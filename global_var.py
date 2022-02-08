# 초단위
work_stime = 8*60*60    # 작업 시작 시각
work_etime = 23*60*60   # 작업 마감 시각
load_time = 15*60       # 로드하는데 걸리는 시간
unload_time = 15*60     # 언로드하는데 걸리는 시간

# 염색체 관련 변수
populationSize = 10     # 한 세대당 염색체 수
n_generations = [100]      # 세대 수

# 트랜스포터와 작업 개수
transporter_num = 100
task_num = [100, 200, 300]

# 초기 위치
init_position = 57

# H1 관련 변수
search_num_h1 = 10

# H2 관련 변수
search_num_h2 = 10

# 유전 할 부모 개수
cross_num = 5

# 돌연변이
mutation = 0.2

# 트랜스포터 스케줄 표 색깔
e_c = ['gray', 'salmon', 'sandybrown', 'bisque', 'gold', 'olivedrab', 'skyblue', 'slategrey', 'slateblue', 'thistle']
w_c = ['black', 'red', 'saddlebrown', 'darkorange', 'darkgoldenrod', 'yellowgreen', 'steelblue', 'darkblue', 'darkslateblue', 'purple']