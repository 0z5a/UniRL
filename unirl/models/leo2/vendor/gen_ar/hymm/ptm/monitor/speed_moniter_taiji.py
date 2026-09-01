import os
import time
import datetime
import re
import numpy as np
# import ipdb
import argparse
import traceback
import logging
import logging.config

my_arg_parser = argparse.ArgumentParser()

my_arg_parser.add_argument("--exp-name", type=str, required=True)
my_arg_parser.add_argument("--log-path", type=str, required=True)
my_arg_parser.add_argument("--batch-size", type=int, required=True)
my_arg_parser.add_argument("--seqlen", type=int, required=True)
my_arg_parser.add_argument("--start-wait-time", type=int, default=0)
my_arg_parser.add_argument("--alias", type=str, required=True)
my_arg_parser.add_argument("--webhook-url", type=str, required=True)
my_arg_parser.add_argument("--monitor-log", type=str, default="")

my_args = my_arg_parser.parse_args()

logger = logging.getLogger("monitor")
print(my_args.monitor_log)
print(my_args.seqlen)

if my_args.monitor_log == "":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
else:
    log_handler = logging.FileHandler(filename=my_args.monitor_log, encoding='utf-8')
    logging.basicConfig(handlers=[log_handler], level=logging.INFO, format='%(asctime)s %(message)s')

logger.info('ready to do start the loop monitor')

# set the training infos and monitor infos
debug_flag = False
enable_phone_call = True
enable_phone_call_jinbao = True
# batchsize = 1152
# seqlen = 8192

batchsize = my_args.batch_size
seqlen = my_args.seqlen

my_alias:str=my_args.alias
webhook_url=my_args.webhook_url

target_dir_rubik=my_args.log_path
cur_datetime = datetime.datetime.now()
current_year = cur_datetime.year
item_keyword = f'{current_year}'

# exp_name="1B_moe_8expert_mqa_4566ffn_300Blr"
exp_name=my_args.exp_name
try:
    os.mkdir(f"/tmp/{exp_name}")
except OSError as error:
    logger.warning(error)
tmp_path = f"/tmp/{exp_name}/tmp_speed_moniter_taiji_log.tail"

# alarm_num=1 for rubik  alarm_num=2 for jakki  alarm_num=3 for jinbao  alarm_num=7 for kaiven
cmd_call_raw ='export http_proxy=http://star-proxy.oa.com:3128;export https_proxy=http://star-proxy.oa.com:3128;/usr/local/zhiyan/agent/bin/report_tool -app_mark 14003_136003_text-to-image-video-base-model -calc_method 0 -instance_mark 172.17.0.1 -tag_set "error_info='+ exp_name + '_{}&alarm_num=4" -metric_val "log_not_report=1"'
#cmd_call_jakki ='export http_proxy=http://9.21.0.122:11113;export https_proxy=http://9.21.0.122:11113;/usr/local/zhiyan/agent/bin/report_tool -app_mark 15618_129227_consistency_monitor -calc_method 0 -instance_mark 172.17.0.1 -tag_set "error_info='+ exp_name + '_{}&alarm_num=2" -metric_val "log_not_report=1"'
#cmd_call_raw_jinbao ='export http_proxy=http://9.21.0.122:11113;export https_proxy=http://9.21.0.122:11113;/usr/local/zhiyan/agent/bin/report_tool -app_mark 15618_129227_consistency_monitor -calc_method 0 -instance_mark 172.17.0.1 -tag_set "error_info='+ exp_name + '_{}&alarm_num=3" -metric_val "log_not_report=1"'
#cmd_call_raw_focus ='export http_proxy=http://9.21.0.122:11113;export https_proxy=http://9.21.0.122:11113;/usr/local/zhiyan/agent/bin/report_tool -app_mark 15618_129227_consistency_monitor -calc_method 0 -instance_mark 172.17.0.1 -tag_set "error_info='+ exp_name + '_{}&alarm_num=8" -metric_val "log_not_report=1"'
#cmd_call_raw_case ='export http_proxy=http://9.21.0.122:11113;export https_proxy=http://9.21.0.122:11113;/usr/local/zhiyan/agent/bin/report_tool -app_mark 15618_129227_consistency_monitor -calc_method 0 -instance_mark 172.17.0.1 -tag_set "error_info='+ exp_name + '_{}&alarm_num=9" -metric_val "log_not_report=1"'
tail_cmd = "tail -n {} {} > {}"
# cmd_begin = "curl 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=9b316776-587d-4dad-b118-3a4005e79367' \
#     -H 'Content-Type: application/json' \
#     -d '{ \"chatid\": \"@all_group\", \
#          \"msgtype\": \"text\",\
#          \"text\": { \"content\": \""
cmd_begin = "export http_proxy=http://star-proxy.oa.com:3128;export https_proxy=http://star-proxy.oa.com:3128;curl '" + webhook_url + "' \
    -H 'Content-Type: application/json' \
    -d '{ \"chatid\": \"@all_group\", \
         \"msgtype\": \"text\",\
         \"text\": { \"content\": \""
cmd_end= "\" } }' "
error_list = ['_overflow_check_and_loss_scale_update', 'AssertionError', 'ECC', 'with error 12, opcode 129', 'with error 5, opcode 129', 'with error 12, opcode 2', 'CUDA error', 'CUDA driver error', 'status=5, opcode 129,',\
              'Got async event : local catastrophic error', 'unspecified launch failure', 'Connection refused', 'No route to host','with error 11, opcode 129','Connection timed out']

#skip_list = ['video_loader', 'get_batch', 'wrong cache data', 'image_loader', "log_validation error", "CUBLAS_STATUS_EXECUTION_FAILED when calling cublasLtMatmul"]
skip_list = ['video_loader', 'get_batch', 'wrong cache data', 'image_loader']

# set the monitor thresholds
# max_loss_threshold = 5.0
# avg_loss_threshold = 1.8
max_loss_threshold = 20.0
avg_loss_threshold = 2
# avg_samps_threshold = 48
# 平均训练日吞吐量，单位 B tokens / day
avg_samps_threshold = 1
throughput_exception_number_upperbound = 1 # 连续吞吐异常次数上限，超过上限就报警
exp_capacity_threshold = 0.5 #0.99
count_sample_threshold = 0
tolerate_threshold = 8 # the threshold of tolerence, if exceeded, then call

# set the log related infos, usually no need to change
sleep_print = 300
sleep_not = 60
sleep_call = 600
count_lines = 3000 #50000
# push_to_bot_every = 3
push_to_bot_every = 1e8

file_not_update_threshold = tolerate_threshold * 60 + sleep_print

# set the temporary params, usually no need to change
last_cur_time = ''
tolerate_num = 0 # the number of tolerated detections
throughput_exception_number = 0 # 连续吞吐异常次数
last_valid_item_name = ''
print_nums = 0

def format_float(val:float):
    return int(val*1e6)/1e6

if my_args.start_wait_time > 0:
    time.sleep(my_args.start_wait_time)

while True:
    # get the log file
    item_list = []
    try:
        item_list = sorted(os.listdir(target_dir_rubik))
    except:
        pass

    if len(item_list) == 0:
        logger.error(f"there is no log files in the directory {target_dir_rubik}")
        time.sleep(sleep_not)
        continue

    candidate_list = []
    for i in range(len(item_list)):
        if item_keyword in item_list[i]:
            candidate_list.append(i)
    if len(candidate_list) == 0:
        logger.error(f"there are not log files in the directory {target_dir_rubik} matching the keyworkd {item_keyword}")
        time.sleep(sleep_not)
        continue

    item_name = item_list[sorted(candidate_list)[-1]]
    # check whether the training has been restared
    if item_name != last_valid_item_name and last_valid_item_name != '':
        cmd_core = f"@{my_alias} \n"
        cmd_core += f"实验{exp_name}, 发现文件切换\n"
        cmd_core += f"        当前训练已重启，此前的文件为{last_valid_item_name}, 当前文件为{item_name}\n"
        cmd_core += f"        请尽快检查是否异常重启!!!"
        cmd_call = cmd_call_raw.format(f'训练疑似重启')
        last_valid_item_name = item_name
        if not debug_flag:
            logger.info(cmd_begin+cmd_core+cmd_end)
            os.system(cmd_begin+cmd_core+cmd_end)
            logger.info(cmd_call)
            os.system(cmd_call)
            time.sleep(sleep_call)
        else:
            logger.info(cmd_core)
        continue

    target_file_rubik = os.path.join(target_dir_rubik, item_name)
    current_time = datetime.datetime.now()
    logger.info(f"current time: {current_time}, {item_name}")
    logger.info(tail_cmd.format(count_lines, target_file_rubik, tmp_path))
    os.system(tail_cmd.format(count_lines, target_file_rubik, tmp_path))
    #os.system(tail_cmd.format(count_lines, target_file_rubik, tmp_path))
    lines = open(tmp_path).readlines()
    print_flag = False

    # init the
    loss_val_list = []
    samples_pers_list = []
    consumed_tokens_list = []
    exp_capacity_rate_list = []
    validation_loss_list = []
    grad_norm_list = []
    nan_nums = 0
    cur_time = ''
    error_line = ''
    iteration_info=""
    call_flag = False
    push_flag = False
    line_no = 0

    latest_train_log_line = 0
    latest_ckpt_log_line = 0
    latest_eval_log_line = 0

    # skip the known words
    for line in lines:
        line_no = line_no + 1

        for skip_error in skip_list:
            if skip_error in line:
                continue

        # check the known error
        for error_code in error_list:
            if error_code in line:
                cmd_call = cmd_call_raw.format(f'硬件故障')
                #cmd_call_jq = cmd_call_jakki.format(f'硬件故障')
                #cmd_call_focus = cmd_call_raw_focus.format(f'硬件故障')
                #cmd_call_case = cmd_call_raw_case.format(f'硬件故障')
                #cmd_call_jinbao = cmd_call_raw_jinbao.format(f'硬件故障')
                call_flag = True
                push_flag = True
                # if get error, directly call and push
                error_line = line
                break

        # gather the statics
        try:
            '''
            if 'norm=' in line:
                try:
                    grad_norm = float(re.split('norm=', line)[1].strip())
                except:
                    grad_norm = 0.0
                grad_norm_list.append(grad_norm)
            '''
            if 'consumed tokens' in line:
                try:
                    loss_val = float(re.split('lm loss:|\| text loss', line)[1].strip())
                except:
                    continue
                if loss_val == 'nan':
                        nan_nums += 1
                else:
                    loss_val_list.append(loss_val)

                try:
                    grad_norm = float(re.split('grad norm:|\| number of skipped iterations', line)[1].strip())
                    #logger.info(f'grad_norm is {grad_norm}')
                except:
                    grad_norm = 0.0
                grad_norm_list.append(grad_norm)

                time_this_step = float(re.split(r'elapsed time this iteration \(ms\):|\| avg elapsed time', line)[1].strip())
                samples_pers = batchsize/time_this_step*1000
                samples_pers_list.append(samples_pers)
                consumed_tokens = float(re.split(r'consumed tokens:|\| elap', line)[1].strip())
                consumed_tokens_list.append(consumed_tokens)

                try:
                    iteration_info  =re.split('\| iteration\s+|\| consumed samples:', line)[1].strip()
                    latest_train_log_line = line_no
                except:
                    iteration_info=""

                print_flag = True

            if '[INFO]' in line:
                cur_time = re.split('\[INFO\]', line)[0]
        except:
            traceback.print_exc()
            pass

        if "checkpoint" in line:
            latest_ckpt_log_line = line_no

        if "Video Sample" in line:
            latest_eval_log_line = line_no

    logger.info(f"after scan {len(lines)} lines, latest training line {latest_train_log_line} checkpoint line {latest_ckpt_log_line} evaluation line {latest_eval_log_line}")
    # if
    if len(loss_val_list) <= count_sample_threshold or not print_flag:
        cmd_core="未检测到log或是log长度不够, 等待1分钟后监测"
        tolerate_num += 1
        if tolerate_num > tolerate_threshold :
            current_timestamp = time.time()
            last_update_time = os.path.getmtime(target_file_rubik)
            gap = current_timestamp - last_update_time
            if int(gap) < file_not_update_threshold:
                cmd_core += f"\n未检测到iteration log:{tolerate_num}, 超过{tolerate_threshold}次, \
                    但是日志在{int(gap)}秒内更新, 小于阈值{file_not_update_threshold}, 应当是在初始化, 只打印日志不电话报警"
                #os.system(cmd_begin+cmd_core+cmd_end)
                logger.info(cmd_core)
                time.sleep(sleep_not)
                continue

            cmd_core=f"未检测到log:{tolerate_num}, 超过{tolerate_threshold}次, 日志不更新超过{int(file_not_update_threshold)}秒, 训练疑似中断！触发电话告警，需要人工排查"
            if not debug_flag:
                os.system(cmd_begin+cmd_core+cmd_end)
                call_flag = True
                logger.info(cmd_core)
                if enable_phone_call and call_flag:
                    cmd_call = cmd_call_raw.format(f'训练疑似卡顿')
                    #cmd_call_jq = cmd_call_jakki.format(f'训练疑似卡顿')
                    #cmd_call_focus = cmd_call_raw_focus.format(f'训练疑似卡顿')
                    #cmd_call_case = cmd_call_raw_case.format(f'训练疑似卡顿')
                    #cmd_call_jinbao = cmd_call_raw_jinbao.format(f'训练疑似卡顿')
                    os.system(cmd_call)
                    #os.system(cmd_call_jq)
                    #os.system(cmd_call_case)
                    #os.system(cmd_call_focus)
                    #if enable_phone_call_jinbao:
                    #    os.system(cmd_call_jinbao)
                    time.sleep(sleep_call)
                else:
                    time.sleep(sleep_not)
            else:
                logger.info(cmd_core)
                time.sleep(sleep_not)
        else:
            logger.info(cmd_core)
            time.sleep(sleep_not)
    else:
        tolerate_num = 0
        latest_log_num = 100
        loss_val = np.mean(loss_val_list[-latest_log_num:])
        max_loss_val = np.max(loss_val_list[-latest_log_num:])

        loss_val_10 = np.mean(loss_val_list[-10:])
        max_loss_val_10 = np.max(loss_val_list[-10:])

        loss_val_20 = np.mean(loss_val_list[-20:])
        max_loss_val_20 = np.max(loss_val_list[-20:])

        loss_val_50 = np.mean(loss_val_list[-50:])
        max_loss_val_50 = np.max(loss_val_list[-50:])

        loss_val_100 = np.mean(loss_val_list[-100:])
        max_loss_val_100 = np.max(loss_val_list[-100:])

        all_loss_val = np.mean(loss_val_list)
        all_max_loss_val = np.max(loss_val_list)

        samples_pers = np.mean(samples_pers_list) * seqlen * 24 * 3600 / 1e9
        min_samples_pers = np.min(samples_pers_list) * seqlen * 24 * 3600 / 1e9
        grad_norms = np.mean(grad_norm_list)
        max_grad_norm = np.max(grad_norm_list)
        consumed_tokens = np.max(consumed_tokens_list)/1e9
        num_samples = len(loss_val_list)
        if iteration_info:
            cmd_core=f"@{my_alias} \n" \
                + f"实验{exp_name}, 监测到 {num_samples} 条log信息 \n" \
                + f"        当前iteration信息 ：{iteration_info} \n" \
                + f"        最近10 条平均 loss：{format_float(loss_val_10)}，最大loss：{format_float(max_loss_val_10)} \n" \
                + f"        最近20 条平均 loss：{format_float(loss_val_20)}，最大loss：{format_float(max_loss_val_20)} \n" \
                + f"        最近50 条平均 loss：{format_float(loss_val_50)}，最大loss：{format_float(max_loss_val_50)} \n" \
                + f"        最近100条平均 loss：{format_float(loss_val_100)}，最大loss：{format_float(max_loss_val_100)} \n" \
                + f"        最近{num_samples}条平均 loss：{format_float(all_loss_val)}，最大loss：{format_float(all_max_loss_val)} \n" \
                + f"        当前平均吞吐：{format_float(samples_pers)} B tokens/d, 最低吞吐：{format_float(min_samples_pers)} B tokens/d \n" \
                + f"        平均梯度norm: {format_float(grad_norms)}, 最大梯度norm: {format_float(max_grad_norm)} \n" \
                + f"        已训练tokens：{format_float(consumed_tokens)}B, nan step的个数{nan_nums} \n "
        else:
            cmd_core=f"@{my_alias} \n" \
                + f"实验{exp_name}, 监测到 {num_samples} 条log信息 \n" \
                + f"        当前iteration信息 ：{iteration_info} \n" \
                + f"        最近10 条平均 loss：{format_float(loss_val_10)}，最大loss：{format_float(max_loss_val_10)} \n" \
                + f"        最近20 条平均 loss：{format_float(loss_val_20)}，最大loss：{format_float(max_loss_val_20)} \n" \
                + f"        最近50 条平均 loss：{format_float(loss_val_50)}，最大loss：{format_float(max_loss_val_50)} \n" \
                + f"        最近100条平均 loss：{format_float(loss_val_100)}，最大loss：{format_float(max_loss_val_100)} \n" \
                + f"        最近{num_samples}条平均 loss：{format_float(all_loss_val)}，最大loss：{format_float(all_max_loss_val)} \n" \
                + f"        当前平均吞吐：{format_float(samples_pers)} B tokens/d, 最低吞吐：{format_float(min_samples_pers)} B tokens/d \n" \
                + f"        平均梯度norm: {format_float(grad_norms)}, 最大梯度norm: {format_float(max_grad_norm)} \n" \
                + f"        已训练tokens：{format_float(consumed_tokens)}B, nan step的个数{nan_nums} \n "
            # cmd_core=f"@{my_alias} 实验{exp_name}, 监测到 {num_samples} 条log信息 \n \
            #     当前iteration信息 ：{iteration_info} \n \
            #     最近10 条平均 loss：{loss_val_10}，最大loss：{max_loss_val_10} \n \
            #     最近20 条平均 loss：{loss_val_20}，最大loss：{max_loss_val_20} \n \
            #     最近50 条平均 loss：{loss_val_50}，最大loss：{max_loss_val_50} \n \
            #     最近100条平均 loss：{loss_val_100}，最大loss：{max_loss_val_100} \n \
            #     最近{num_samples}条平均 loss：{all_loss_val}，最大loss：{all_max_loss_val} \n \
            #     当前平均吞吐：{samples_pers} B tokens/d, 最低吞吐：{min_samples_pers} B tokens/d \n \
            #     平均负载率:{exp_capacity_rate}%, 最低负载率: {min_exp_capacity_rate}% \n \
            #     平均梯度norm: {grad_norms}, 最大梯度norm: {max_grad_norm}, \n \
            #     已训练tokens：{consumed_tokens}B nan step的个数{nan_nums} \n "
        #if len(validation_loss_list) > 0:
        #    cmd_core += "\n监测到 {} 条validation log信息，当前 validation loss： {:.3f}".format(len(validation_loss_list),np.mean(validation_loss_list))

        # check whether the statics are within the thresholds
        if loss_val >= avg_loss_threshold:
            cmd_core += f"\n当前训练loss均值异常！！！！，当前值{loss_val}, 设置阈值{avg_loss_threshold}"
            cmd_call = cmd_call_raw.format(f"loss异常")
            #cmd_call_jq = cmd_call_jakki.format(f"loss异常")
            #cmd_call_focus = cmd_call_raw_focus.format(f'loss异常')
            #cmd_call_case = cmd_call_raw_case.format(f'loss异常')
            #cmd_call_jinbao = cmd_call_raw_jinbao.format(f"loss异常")
            push_flag = True
            call_flag = True

        if max_loss_val >= max_loss_threshold:
            cmd_core += f"\n当前训练loss极大值偏高！！！！，当前值{max_loss_val}, 设置阈值{max_loss_threshold}"
            cmd_call = cmd_call_raw.format(f"loss异常")
            #cmd_call_jq = cmd_call_jakki.format(f"loss异常")
            #cmd_call_focus = cmd_call_raw_focus.format(f'loss异常')
            #cmd_call_case = cmd_call_raw_case.format(f'loss异常')
            #cmd_call_jinbao = cmd_call_raw_jinbao.format(f"loss异常")
            push_flag = True
            call_flag = True

        if samples_pers <= avg_samps_threshold:
            throughput_exception_number += 1
            if throughput_exception_number > throughput_exception_number_upperbound:
                cmd_core += f"\n当前训练吞吐异常！！！！，当前值{samples_pers}, 设置阈值{avg_samps_threshold}"
                cmd_call = cmd_call_raw.format(f"吞吐异常")
                #cmd_call_jq = cmd_call_jakki.format(f"吞吐异常")
                #cmd_call_focus = cmd_call_raw_focus.format(f'吞吐异常')
                #cmd_call_case = cmd_call_raw_case.format(f'吞吐异常')
                #cmd_call_jinbao = cmd_call_raw_jinbao.format(f"吞吐异常")
                push_flag = True
                call_flag = True
        else:
            throughput_exception_number = 0

        if nan_nums > 0:
            cmd_core += f"\n当前训练loss异常出现nan！！！！，当前值{nan_nums}"
            cmd_call = cmd_call_raw.format(f"loss NaN")
            #cmd_call_jq = cmd_call_jakki.format(f"loss NaN")
            #cmd_call_focus = cmd_call_raw_focus.format(f'loss NaN')
            #cmd_call_case = cmd_call_raw_case.format(f'loss NaN')
            #cmd_call_jinbao = cmd_call_raw_jinbao.format(f"loss NaN")
            push_flag = True
            push_flag = True
            call_flag = True


        # check whether the training has been restared
        if item_name != last_valid_item_name and last_valid_item_name != '':
            push_flag = True
            cmd_core += f"\n当前训练已重启，此前的文件为{last_valid_item_name}, 当前文件为{item_name}"
        last_valid_item_name = item_name

        # check whether the training has been stucked
        logger.info(cur_time)
        cur_time = re.split('\[|\]',cur_time)[1]
        cmd_core += f'\n最后一条log的打印时间为{cur_time}'
        if cur_time == last_cur_time:
            current_timestamp = time.time()
            last_update_time = os.path.getmtime(target_file_rubik)
            gap = current_timestamp - last_update_time
            if int(gap) < file_not_update_threshold:
                cmd_core += f"\n最近一次检测训练iteration数据未输出, 但是日志在{int(gap)}秒内更新, 小于阈值{file_not_update_threshold}"
                if latest_eval_log_line > latest_ckpt_log_line and latest_eval_log_line > latest_train_log_line:
                    cmd_core += f"\n最新的日志显示正在进行evaluation, 只打印日志不电话报警"
                    push_flag = False
                    call_flag = False
                elif latest_ckpt_log_line > latest_eval_log_line and latest_ckpt_log_line > latest_train_log_line:
                    cmd_core += f"\n最新的日志关显示正在进行checkpoint, 只打印日志不电话报警"
                    push_flag = False
                    call_flag = False
                else:
                    cmd_core += f"\n当前日志正在输出但阶段未知, 请联系研发排查是否正常"
                    cmd_call = cmd_call_raw.format('训练疑似卡顿')
                    push_flag = True
                    call_flag = True
            else:
                cmd_core += f"\n最近一次检测训练iteration数据未输出, 同时日志在{int(gap)}秒内都没更新, 大于等于阈值{file_not_update_threshold}"
                cmd_core += '\n###############################训练疑似卡顿，需要尽快查看！#############################'
                cmd_call = cmd_call_raw.format('训练疑似卡顿')
                #cmd_call_jq = cmd_call_jakki.format('训练疑似卡顿')
                #cmd_call_focus = cmd_call_raw_focus.format(f'训练疑似卡顿')
                #cmd_call_case = cmd_call_raw_case.format(f'训练疑似卡顿')
                #cmd_call_jinbao = cmd_call_raw_jinbao.format('训练疑似卡顿')
                push_flag = True
                call_flag = True
        last_cur_time = cur_time

        # call or push to the bot
        if not debug_flag:
            logger.info(cmd_begin+cmd_core+cmd_end)
            if print_nums % push_to_bot_every==0 or push_flag:
                if error_line != '':
                    cmd_core += f'\n报错行信息为{error_line}'
                os.system(cmd_begin+cmd_core+cmd_end)
                print_nums = 1
            print_nums += 1
            if enable_phone_call and call_flag:
                os.system(cmd_call)
                #os.system(cmd_call_jq)
                #os.system(cmd_call_focus)
                #os.system(cmd_call_case)
                # if enable_phone_call_jinbao:
                #    os.system(cmd_call_jinbao)
                time.sleep(sleep_call)
        else:
            logger.info(cmd_core)
        time.sleep(sleep_print)

    if iteration_info:
        try:
            # 9487/   11920
            iter_info_list = iteration_info.split("/")
            if iter_info_list and len(iter_info_list) == 2:
                if int(iter_info_list[0].strip()) == int(iter_info_list[1].strip()):
                    break
        except:
            pass
