# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import itertools
import sys
import time
import pandas as pd
import numpy as np
from omegaconf import DictConfig
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from model.utils.utils import *
from model.utils import datasets
from model.utils.iterators import EpochBatchIterator
from model import CombinedSpatiotemporalModel5
from model.temporal import NeuralPointProcess
from model.temporal.neural import ACTFNS as TPP_ACTFNS
from model.transformer.Models import Encoder
from setproctitle import setproctitle

torch.backends.cudnn.benchmark = True


def validate(model, test_loader, t0, t1, device):

    model.eval()

    msle_meter = AverageMeter()
    smape_meter = AverageMeter()
    loglik_meter = AverageMeter()
    metric = Metric()

    with torch.no_grad():
        for batch in test_loader:
            event_times, spatial_events, input_mask, label, seq, cum_seq, id = map(lambda x: cast(x, device), batch)
            pre_label, loglik = model(event_times, spatial_events, input_mask, t0, t1, label, seq, cum_seq)

            metric.update(pre_label.squeeze(-1).clone().detach().tolist(), label.squeeze(-1).clone().detach().tolist(),
                          id.squeeze(-1).clone().detach().tolist(), input_mask.sum(-1).squeeze(-1).clone().detach().tolist())
            msle_loss = get_msle(pre_label, label)
            smape = get_smape(pre_label, label)
            msle_meter.update(msle_loss.mean().item(), input_mask.shape[0])
            smape_meter.update(smape.mean().item(), input_mask.shape[0])
            loglik_meter.update(loglik.item(), input_mask.shape[0])
        all_metric, pred_csv = metric.calculate_metric()

    model.train()

    return msle_meter.avg, smape_meter.avg, loglik_meter.avg, all_metric, pred_csv


def _main(rank, world_size, args, savepath, logger):

    if rank == 0:
        logger.info(args)
        logger.info(f"Saving to {savepath}")
        tb_writer = SummaryWriter(os.path.join(savepath, "tb_logdir"))

    device = torch.device(f'cuda:{rank:d}' if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        if device.type == 'cuda':
            logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info('{} \t Memory: {:.2f}GB'.format(props.name, props.total_memory / (1024**3)))
        else:
            logger.info('WARNING: Using device {}'.format(device))

    t0, t1 = map(lambda x: cast(x, device), get_t0_t1(args.t1, args))
    print('t0:', t0, 't1:', t1)

    x_dim = args.cg_emb_dim

    train_set = datasets.CascadeData(args, split="train", observation_time=args.observation_time,
                                     seq_len=args.max_seq, x_dim=x_dim)
    val_set = datasets.CascadeData(args, split="val", observation_time=args.observation_time,
                                   seq_len=args.max_seq, x_dim=x_dim)
    test_set = datasets.CascadeData(args, split="test", observation_time=args.observation_time,
                                    seq_len=args.max_seq, x_dim=x_dim)

    train_epoch_iter = EpochBatchIterator(
        dataset=train_set,
        collate_fn=datasets.spatiotemporal_events_collate_fn,
        batch_sampler=train_set.batch_by_size(args.max_events),
        seed=args.seed + rank,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.test_bsz,
        shuffle=False,
        collate_fn=datasets.spatiotemporal_events_collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_bsz,
        shuffle=False,
        collate_fn=datasets.spatiotemporal_events_collate_fn,
    )

    if rank == 0:
        logger.info(f"{len(train_set)} training examples, {len(val_set)} val examples, {len(test_set)} test examples")


    if args.model == 'ode_diff_increase':
        tpp_hidden_dims = list(map(int, args.tpp_hdims.split("-")))
        tpp_model = NeuralPointProcess(
            cond_dim=x_dim, hidden_dims=tpp_hidden_dims, cond=args.tpp_cond, style=args.tpp_style,
            actfn=args.tpp_actfn,
            otreg_strength=args.tpp_otreg_strength, tol=args.tol, solver=args.ode_solver)
        encoder_model = Encoder(d_model=args.cg_emb_dim, d_inner=args.d_inner_hid, n_layers=args.n_layers,
                                n_head=args.n_head, d_k=args.d_k, d_v=args.d_v, dropout=args.dropout, device=device,
                                t_scale=args.t_scale, global_attention=args.global_attention)
        model = CombinedSpatiotemporalModel5(tpp_model, encoder_model, args).to(device)



    params = []
    attn_params = []
    for name, p in model.named_parameters():
        if "self_attns" in name:
            attn_params.append(p)
        else:
            params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": params},
        {"params": attn_params}
    ], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))

    if rank == 0:
        ema = ExponentialMovingAverage(model)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if rank == 0:
        logger.info(model)

    begin_itr = 0
    checkpt_path = os.path.join(savepath, "model.pth")
    csv_path = os.path.join(savepath, "prediction.csv")
    if os.path.exists(checkpt_path):
        # Restart from checkpoint if run is a restart.
        if rank == 0:
            logger.info(f"Resuming checkpoint from {checkpt_path}")
        checkpt = torch.load(checkpt_path, "cpu")
        model.module.load_state_dict(checkpt["state_dict"])
        optimizer.load_state_dict(checkpt["optim_state_dict"])
        begin_itr = checkpt["itr"] + 1

    elif args.resume:
        # Check the resume flag if run is new.
        if rank == 0:
            logger.info(f"Resuming model from {args.resume}")
        checkpt = torch.load(args.resume, "cpu")
        model.module.load_state_dict(checkpt["state_dict"])
        optimizer.load_state_dict(checkpt["optim_state_dict"])
        begin_itr = checkpt["itr"] + 1

    msle_meter = AverageMeter()
    smape_meter = AverageMeter()
    loglik_meter = AverageMeter()
    best_val_msle = 9999
    best_test_msle = 9999
    best_val_smape = 9999
    best_test_smape = 9999
    patience = 0
    best_all_metric = None

    model.train()
    start_time = time.time()
    iteration_counter = itertools.count(begin_itr)
    begin_epoch = begin_itr // len(train_epoch_iter)
    args.num_iterations = len(train_epoch_iter) * args.max_epoch

    for epoch in range(begin_epoch, args.max_epoch):
        batch_iter = train_epoch_iter.next_epoch_itr(shuffle=True)

        for batch in batch_iter:
            itr = next(iteration_counter)

            optimizer.zero_grad()

            event_times, spatial_events, input_mask, label, seq, cum_seq, _ = map(lambda x: cast(x, device), batch)
            N, T = input_mask.shape
            num_events = input_mask.sum()

            if num_events == 0:
                raise RuntimeError("Got batch with no observations.")

            pre_label, loglik = model(event_times, spatial_events, input_mask, t0, t1, label, seq, cum_seq)

            msle_loss = get_msle(pre_label, label)
            msle_meter.update(msle_loss.mean().item(), N)
            smape = get_smape(pre_label, label)
            smape_meter.update(smape.mean().item(), N)

            loglik_meter.update(loglik.item())

            # if args.loss:
            loss = msle_loss.mean() + loglik
            loss.backward()

            # Set learning rate
            total_itrs = math.ceil(args.num_iterations / len(train_epoch_iter)) * len(train_epoch_iter)
            lr = learning_rate_schedule(itr, args.warmup_itrs, args.lr, total_itrs)
            set_learning_rate(optimizer, lr)

            optimizer.step()

            if rank == 0:
                if itr > 0.8 * args.num_iterations:
                    ema.apply()
                else:
                    ema.apply(decay=0.0)

            if rank == 0:
                tb_writer.add_scalar("train/lr", lr, itr)
                tb_writer.add_scalar("train/msle", msle_loss.mean().item(), itr)
                tb_writer.add_scalar("train/smape", smape.mean().item(), itr)
                tb_writer.add_scalar("train/loglik", loglik.item(), itr)

            if itr % args.logfreq == 0:
                elapsed_time = time.time() - start_time

                # Sum memory usage across devices.
                mem = torch.tensor(memory_usage_psutil()).float().to(device)
                dist.all_reduce(mem, op=dist.ReduceOp.SUM)

                if rank == 0:
                    logger.info(
                        f"Iter {itr} | Epoch {epoch} | LR {lr:.5f} | Time {elapsed_time:.1f}"
                        f" | msle {msle_meter.val:.4f}({msle_meter.avg:.4f})"
                        f" | smape {smape_meter.val:.4f}({smape_meter.avg:.4f})"
                        f" | loglik {loglik_meter.val:.4f}({loglik_meter.avg:.4f})"
                        # f" | NFE {nfe.item()}"
                        f" | Mem {mem.item():.2f} MB")

                    # tb_writer.add_scalar("train/nfe", nfe, itr)
                    tb_writer.add_scalar("train/time_per_itr", elapsed_time / args.logfreq, itr)

                start_time = time.time()

        if rank == 0:
            pre_label_val, smape_val, val_loglik, val_all_metric, _ = validate(model, val_loader, t0, t1, device)
            pre_label_test, smape_test, test_loglik, test_all_metric, pred_csv = validate(model, test_loader, t0, t1, device)

            if pre_label_val < best_val_msle:
                patience = 0
                best_val_msle = pre_label_val
                best_test_msle = pre_label_test

                best_val_smape = smape_val
                best_test_smape = smape_test

                best_all_metric = test_all_metric

                torch.save({
                    "itr": itr,
                    "state_dict": model.module.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "ema_parmas": ema.ema_params,
                }, checkpt_path)

                df = pd.DataFrame(pred_csv)
                df.to_csv(csv_path)

                # ema.swap()
                logger.info(
                    f"[Test] Iter {itr} | Epoch {epoch}| Val msle {pre_label_val:.4f} | Val smape {smape_val:.4f} | Val loglik {val_loglik:.4f}"
                    f" | Test msle {pre_label_test:.4f} | Test smape {smape_test:.4f}| Test loglik {test_loglik:.4f}"
                    f" | Best Test msle {best_test_msle:.4f} | Best Val msle {best_val_msle:.4f}"
                    f" | Best Test smape {best_test_smape:.4f} | Best Val smape {best_val_smape:.4f}"
                    f" | Other Best Test metrics {best_all_metric}")

                tb_writer.add_scalar("val/msle", pre_label_val, itr)
                tb_writer.add_scalar("test/msle", pre_label_test, itr)
                tb_writer.add_scalar("val/smape", smape_val, itr)
                tb_writer.add_scalar("test/smape", smape_test, itr)

            else:
                patience += 1
                logger.info(
                    f"[Test] Iter {itr} | Epoch {epoch}| Val msle {pre_label_val:.4f} | Val smape {smape_val:.4f} | Val loglik {val_loglik:.4f}"
                    f" | Test msle {pre_label_test:.4f} | Test smape {smape_test:.4f}| Test loglik {test_loglik:.4f}"
                    f" | Best Test msle {best_test_msle:.4f} | Best Val msle {best_val_msle:.4f}"
                    f" | Best Test smape {best_test_smape:.4f} | Best Val smape {best_val_smape:.4f}"
                    f" | Other Best Test metrics {best_all_metric}")

                tb_writer.add_scalar("val/msle", pre_label_val, itr)
                tb_writer.add_scalar("test/msle", pre_label_test, itr)
                tb_writer.add_scalar("val/smape", smape_val, itr)
                tb_writer.add_scalar("test/smape", smape_test, itr)
                if patience > 15:
                    logger.info("Early Stop!")
                    # ema.swap()
                    break

        msle_meter.reset()
        smape_meter.reset()
        loglik_meter.reset()

        start_time = time.time()

    if rank == 0:
        tb_writer.close()


def main(rank, world_size, args, savepath):
    setup(rank, world_size, args.port)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    logger = get_logger(os.path.join(savepath, "logs.txt"))

    try:
        _main(rank, world_size, args, savepath, logger)
    except:
        import traceback
        logger.error(traceback.format_exc())
        raise
    cleanup()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="aps")
    parser.add_argument("--observation_time", type=str, default="1095")
    parser.add_argument("--prediction_time", type=str, default="7305")
    parser.add_argument("--max_events", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--interval_num", type=int, default=10)
    parser.add_argument("--tpp_hdims", type=str, default="64")
    parser.add_argument("--loss_type", type=str, choices=["l1", "l2", "Euclid"], default="l2")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling_timesteps", type=int, default=100)
    parser.add_argument("--objective", type=str, choices=["pred_noise", "pred_x0", "pred_v"], default="pred_x0")
    parser.add_argument("--loss_scale", type=float, default=0.5)
    parser.add_argument("--FC", type=int, default=0)
    parser.add_argument("--ode_solver", type=str, choices=["dopri5", "dopri8", "bosh3", "adaptive_heun", "euler",
                                                           "midpoint", "rk4", "explicit_adams", "implicit_adams",
                                                           "scipy_solver"], default="dopri5")
    parser.add_argument("--max_seq", type=str, default="100")
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, choices=['ode_diff_increase'],
                        default="ode_diff_increase")

    parser.add_argument("--cg_emb_dim", type=int, default=80)
    parser.add_argument("--t1", type=int, default=1)
    #transformer
    parser.add_argument('--d_model', type=int, default=80)
    parser.add_argument('--d_inner_hid', type=int, default=64)
    parser.add_argument('--d_k', type=int, default=32)
    parser.add_argument('--d_v', type=int, default=32)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--t_scale', type=float, default=1)
    parser.add_argument("--global_attention", action="store_true")

    parser.add_argument("--actfn", type=str, default="swish")
    parser.add_argument("--tpp_actfn", type=str, choices=TPP_ACTFNS.keys(), default="softplus")
    parser.add_argument("--layer_type", type=str, choices=["concat", "concatsquash"], default="concat")

    parser.add_argument("--tpp_nocond", action="store_false", dest='tpp_cond')
    parser.add_argument("--tpp_style", type=str, choices=["split", "simple", "gru"], default="gru")
    parser.add_argument("--no_share_hidden", action="store_false", dest='share_hidden')
    parser.add_argument("--solve_reverse", action="store_true")
    parser.add_argument("--l2_attn", action="store_true")
    parser.add_argument("--naive_hutch", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--otreg_strength", type=float, default=1e-4)
    parser.add_argument("--tpp_otreg_strength", type=float, default=1e-4)

    parser.add_argument("--warmup_itrs", type=int, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--gradclip", type=float, default=0)
    parser.add_argument("--test_bsz", type=int, default=320)

    parser.add_argument("--experiment_dir", type=str, default="experiments")
    parser.add_argument("--experiment_id", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--logfreq", type=int, default=10)
    parser.add_argument("--testfreq", type=int, default=1000)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    if args.port is None:
        args.port = int(np.random.randint(10000, 20000))

    if args.experiment_id is None:
        args.experiment_id = time.strftime("%Y%m%d_%H%M%S")

    setproctitle(f"xj_concat_{args.data}_{args.observation_time}_{args.max_seq}")

    experiment_name = f"{args.model}"
    experiment_name += f"_h{args.tpp_hdims}_im{args.interval_num}_T{args.timesteps}_ls{args.loss_scale}"
    experiment_name += f"_lr{args.lr}"
    experiment_name += f"_bsz{args.max_events}_FC{args.FC}"

    savepath = os.path.join(args.experiment_dir, args.data + '_t' + args.observation_time + '_s' + args.max_seq, experiment_name)

    # Top-level logger for logging exceptions into the log file.
    makedirs(savepath)
    logger = get_logger(os.path.join(savepath, "logs.txt"))

    if args.gradclip == 0:
        args.gradclip = 1e10

    try:
        mp.set_start_method("forkserver")
        mp.spawn(main,
                 args=(args.ngpus, args, savepath),
                 nprocs=args.ngpus,
                 join=True)
    except Exception:
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
