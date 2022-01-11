from global_var import e_c, w_c
import matplotlib.pyplot as plt
import matplotlib.dates as md
import pandas as pd


def transporter_schedule_view(trans_manager, title):
    fig = plt.figure(figsize=(18, 8))  ## 캔버스 생성
    fig.set_facecolor('white')  ## 캔버스 색상 설정
    ax = fig.add_subplot()  ## 그림 뼈대(프레임) 생성

    for i, trans in enumerate(trans_manager.t_list):
        df = dataframe_create(trans)
        print("df:", df)
        c_i = i % len(e_c)
        for empty in df['empty']:
            y = [i, i]
            ax.plot(empty, y, color=e_c[c_i], marker='o', linewidth=4.0)

        for work in df['work']:
            y = [i, i]
            ax.plot(work, y, color=w_c[c_i], marker='o', linewidth=2.0)

        # 가로선
        ax.axhline(i+0.5, 0, 1, color='r', linewidth=1)

    # ax.legend() #범레
    xfmt = md.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    fig.autofmt_xdate()
    plt.title('Transporter Schedule_{}'.format(title), fontsize=20)
    plt.xlabel('Time(HH:MM:SS)')
    plt.ylabel('Transporter Number')
    plt.show()

# 데이터 프레임 생성
def dataframe_create(trans):
    df = {"empty": [], "work": []}
    empty_list = []
    work_list = []
    for i in range(len(trans.task)):
        empty, work = trans.t_time(i)

        empty_list.append(empty)
        work_list.append(work)

    df["empty"] = empty_list
    df["work"] = work_list

    return df






