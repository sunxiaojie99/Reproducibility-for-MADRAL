from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import os
import shutil


# 加载日志数据
skip = 10  # 每多少步输出一次
forward_skip = 0  # 前多少步不输出
name = '1e-4-multi-predict-1-6-12-shared-dynamic-a01-concat'


data_dir = '/path/attempt-code/output/'+name + '/tboard'
save_dir = 'board/'
print(save_dir)
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)


def PltSctr(ax, a, b):
    ax.scatter(a, b, color='b')
    ax.annotate((a, b), xy=(a, b), xytext=(a, b), fontweight="bold")


ea = event_accumulator.EventAccumulator(data_dir)
ea.Reload()
all_keys = ea.scalars.Keys()
print(all_keys)
for key in all_keys:
    val_acc = ea.scalars.Items(key)

    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(111)
    step_list = []
    value_list = []
    min_step = 0
    min_value = 100
    max_step = 0
    max_value = -100
    for i in val_acc[forward_skip:]:
        value = round(i.value, 2)
        step = i.step
        if step % skip != 0:
            continue
        step_list.append(step)
        value_list.append(value)
        if i.value < min_value:
            min_value = value
            min_step = step
        if value > max_value:
            max_value = value
            max_step = step

    ax1.plot(step_list, value_list, 'lightblue', label=None)
    ax1.set_xlabel("step")
    ax1.set_ylabel(key)
    ax1.legend(loc='lower right')
    # plt.text(min_step, min_value, '({},{})'.format(
    #     min_step, min_value), color="r")
    PltSctr(ax1, min_step, min_value)
    PltSctr(ax1, max_step, max_value)
    # plt.savefig(data_dir+'/'+key+'.pdf', bbox_inches='tight')
    print(key, len(val_acc))
    print('min_step:{}, min_value:{}'.format(min_step, min_value))
    plt.title(name)
    plt.savefig(save_dir + key+'.pdf', bbox_inches='tight')
    # plt.show()
