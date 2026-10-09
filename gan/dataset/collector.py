from gan.utils import Builders
import torch

class Collector:
    def __init__(self,seq_len ,n_actions):
        super().__init__()
        self.blds = Builders(0,max_len=seq_len,n_actions=n_actions)
        self.blds_bak = Builders(0,max_len=seq_len,n_actions=n_actions)
        self.seq_len = seq_len
        self.n_actions = n_actions

    def reset(self,data,target,metric, use_simple_metric=True):
        # Incremental update backup pool
        prev_bak_size = self.blds_bak.batch_size
        new_exprs_size = self.blds.batch_size

        # Record whether the new expression has been evaluated
        new_exprs_evaluated = self.blds.examined

        self.blds_bak += self.blds  # Use optimized __add__ method

        print(f'[INFO] Backup pool incremental update: {prev_bak_size} + {new_exprs_size} = {self.blds_bak.batch_size} expressions')

        if self.blds_bak.batch_size > 0:
            import time
            start_time = time.time()

            # Check whether evaluation is required
            if not self.blds_bak.examined:
                # Determine the evaluation method and metric
                if use_simple_metric:
                    # Use simplified AFF-style metric evaluation backup pool
                    def simple_backup_metric(fct, tgt):
                        from alphagen.utils.correlation import batch_ret, batch_pearsonr
                        import torch
                        import numpy as np

                        ret = batch_ret(fct, tgt)
                        ic = batch_pearsonr(fct, tgt)

                        ic_mean = ic.mean().abs().item()
                        icir = (ic_mean/ic.std()).item()
                        ret_mean = ret.mean().abs().item()
                        ret_ir = (ret_mean/ret.std()).item()
                        sharpe = ((ret_mean- 0.03/252)/ret.std() * np.sqrt(252)).item()

                        def invalid_to_zero(x):
                            if not np.isfinite(x):
                                return 0.
                            else:
                                return max(x,0.)

                        multi_score = {'ic':ic_mean,'icir':icir,'ret':ret_mean,'sharpe':sharpe,'retir':ret_ir}
                        multi_score = {k:invalid_to_zero(v) for k,v in multi_score.items()}
                        score = multi_score['ic']

                        # Basic data quality check
                        if torch.isfinite(fct[0]).sum()/torch.isfinite(tgt[0]).sum() < 0.8:
                            score = 0.
                        elif len(torch.unique(fct[0])) / len(torch.unique(tgt[0])) < 0.01:
                            score = 0.

                        return {
                            'score': score,
                            'ret': ret.detach().cpu().numpy(),
                            'multi_score': multi_score
                        }

                    eval_metric = simple_backup_metric
                    metric_desc = "(Simplified metric)"
                else:
                    eval_metric = metric
                    metric_desc = ""

                # Determine whether it is incremental evaluation
                if hasattr(self.blds_bak, '_needs_evaluation_mask') and self.blds_bak._needs_evaluation_mask is not None:
                    needs_eval_count = sum(self.blds_bak._needs_evaluation_mask)
                    print(f'[INFO] Backup pool incremental evaluation: evaluation only {needs_eval_count}/{self.blds_bak.batch_size} new expressions')
                else:
                    print(f'[INFO] Complete evaluation of backup pool: {self.blds_bak.batch_size} expressions')

                self.blds_bak.evaluate(data,target,eval_metric,verbose=True)

                end_time = time.time()
                eval_time = end_time - start_time
                speed = self.blds_bak.batch_size / eval_time if eval_time > 0 else 0
                print(f'[INFO] Backup pool evaluation completed {metric_desc}, time {eval_time:.1f} s, speed {speed:.1f} exprs/s')
            else:
                print(f'[INFO] The backup pool does not need to be re-evaluated, it has {self.blds_bak.batch_size} evaluated expressions')

        # Reset the current expression set
        self.blds = Builders(0,max_len=self.seq_len,n_actions=self.n_actions)

    def collect(self,netG,netM,z,reset_net = False,random_method=None):
        netG.eval()
        with torch.no_grad():
            if reset_net:
                netG.initialize_parameters()
            netG.eval()
            z = random_method(z)
            netG_output = netG(z)

            # Handling the output formats of different generators
            if isinstance(netG_output, tuple):
                # LSTM generator: return ((onehot_s, logit_s, mask_s), builders)
                (onehot_s, logit_s, mask_s), blds = netG_output
                # The LSTM generator already contains syntax constraints and returns directly to the builder
                return blds
            else:
                # DCGAN/CNN generator: return logits
                logit_raw = netG_output
                masked_x, masks, blds = netM(logit_raw)
                return blds

    def collect_randomly(self,z,netM):
        logit_raw = torch.randn([z.shape[0],self.seq_len,self.n_actions])
        print('logit_raw',logit_raw.shape)
        masked_x,masks,blds= netM(logit_raw)
        return blds


    def collect_target_num(self,netG,netM,z,
                           data,target,metric = None,
                           target_num=1000,reset_net =False,
                           drop_invalid=False,
                           randomly = False,
                           random_method = lambda x:x.normal_(),
                           max_iter = 1000,
                           ):
        from tqdm import tqdm

        cnt = 0
        iter_num = 0

        # Create a progress bar to display the collection progress
        pbar = tqdm(total=target_num, desc=f"Collection expression {'(random)' if randomly else ''}",
                   unit="exprs", ncols=80)

        while self.blds.batch_size <= target_num:
            if randomly:
                builders = self.collect_randomly(z,netM)
            else:
                builders = self.collect(netG,netM,z,reset_net=reset_net,random_method=random_method)
            if drop_invalid:
                builders.drop_invalid()

            prev_size = self.blds.batch_size
            self.blds += builders
            self.blds.drop_duplicated()
            new_size = self.blds.batch_size

            # Update progress bar
            if new_size > prev_size:
                progress = new_size - prev_size
                # Ensure that the overall target is not exceeded
                remaining = max(0, target_num - prev_size)
                actual_progress = min(progress, remaining)
                if actual_progress > 0:
                    pbar.update(actual_progress)
                pbar.set_postfix({
                    'iter': iter_num + 1,
                    'batch': builders.batch_size,
                    'total': self.blds.batch_size
                })

            cnt += 1
            iter_num += 1

            if iter_num > max_iter and max_iter > 0:
                print(f'Reach the maximum number of iterations:{max_iter}, stop collecting')
                break

        # Ensure that the progress bar displays the final status
        final_collected = min(self.blds.batch_size, target_num)
        if pbar.n < final_collected:
            pbar.update(final_collected - pbar.n)
        pbar.close()

        self.blds.drop_duplicated()
        print(f"Collection completed, a total of {self.blds.batch_size} expressions")

        # Evaluate the collected expressions and display a progress bar
        if not self.blds.examined:
            print(f"[INFO] Evaluate the collected expression ({self.blds.batch_size})...")
            self.blds.evaluate(data,target,metric,verbose=True)
        else:
            print(f"[INFO] Skip repeated evaluation:{self.blds.batch_size} expressions have been evaluated")
        return
    @property
    def blds_list(self):
        return [self.blds_bak,self.blds]




# class Collector_2:
#     def __init__(self):
#         super().__init__()
#         self.blds = Builders(0)
#         self.blds_bak = Builders(0)

#     def reset(self,data,target,metric):
#         self.blds_bak += self.blds
#         print('Reset bak_len:',self.blds_bak.batch_size)
#         self.blds_bak.evaluate(data,target,metric)
#         self.blds = Builders(0)

#     def collect(self,netG,netM,z,reset_net = False,random_method=None):
#         with torch.no_grad():
#             if reset_net:
#                 netG.initialize_parameters()
#             netG.eval()
#             random_method(z)
#             logit_raw = netG(z)
#             masked_x,masks,blds= netM(logit_raw)
#             return blds

#     def collect_randomly(self,z,netM):
#         logit_raw = torch.randn([z.shape[0],self.seq_len,self.n_actions])
#         masked_x,masks,blds= netM(logit_raw)
#         return blds
#     def collect_target_num(self,netG,netM,z,data,target,target_num=1000,reset_net =False,drop_invalid=False,
#                            random_method = lambda x:x.normal_(),metric = None,randomly=False):

#         cnt = 0

#         iter_num = 0
#         while self.blds.batch_size<=target_num:
#             if randomly:
#                 builders = self.collect_randomly(z,netM)
#             else:
#                 builders = self.collect(netG,netM,z,reset_net=reset_net,random_method=random_method)
#             if drop_invalid:
#                 builders.drop_invalid()
#             self.blds += builders
#             self.blds.drop_duplicated()
#             cnt += 1
#             iter_num += 1
#             if iter_num%10==0:
#                 print(f"cnt:{cnt} builders_len:{builders.batch_size},all_len:{self.blds.batch_size}")
#             if iter_num>100:
#                 print('iter_num>100')
#                 break

#         self.blds.drop_duplicated()
#         print(self.blds.batch_size)
#         self.blds.evaluate(data,target,metric)
#         return
#     @property
#     def blds_list(self):
#         return [self.blds_bak,self.blds]
