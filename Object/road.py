import math

class Road:
    def __init__(self, start, end, width):
        # self.no = no
        self.s = start      # 시작 위치 클래스
        self.e = end        # 도착 위치 클래스
        self.w = width      # 넓이
        self.d = math.sqrt((start.x - end.x)**2 + (start.y - end.y)**2)     # 길이
        self.s_e = {}   # 시작 --> 도착에 있는 트랜스포터
        self.e_s = {}       # 도착 --> 시작에 있는 트랜스포터
        self.in_s_e = None      # 시작 순
        self.out_s_e = None     # 끝 순
        self.in_e_s = None      # 시작 순
        self.out_e_s = None     # 끝 순

    # # 트랜스포터 사용 시각 추가
    # def add_(self, trans, data):
    #     start_time = data[0]
    #     end_time = data[1]
    #     start_node = data[2]
    #     end_node = data[3]
    #
    #     if start_node == self.s.no:
    #         if trans is not self.s_e:
    #             self.s_e[trans] = [(start_time, end_time)]
    #         else:
    #             self.s_e[trans] = self.s_e[trans] + [(start_time, end_time)]
    #     else:
    #         if trans is not self.e_s:
    #             self.e_s[trans] = [(start_time, end_time)]
    #         else:
    #             self.e_s[trans] = self.e_s[trans] + [(start_time, end_time)]
    #
    # # 시간 순 정렬
    # def sort_dict(self):
    #     if len(self.s_e) > 0:
    #         self.in_s_e = sorted(self.s_e.items(), key=lambda item: item[1][0][0])
    #         self.out_s_e = sorted(self.s_e.items(), key=lambda item: item[1][0][1])
    #         self.in_s_e = dict(self.in_s_e)
    #         self.out_s_e = dict(self.out_s_e)
    #         self.out_s_e.values()
    #
    #
    #     if len(self.e_s) > 0:
    #         self.in_e_s = sorted(self.e_s.items(), key=lambda item: item[1][0][0])
    #         self.out_e_s = sorted(self.e_s.items(), key=lambda item: item[1][0][1])
    #         self.in_e_s = dict(self.in_e_s)
    #         self.out_e_s = dict(self.out_e_s)

    # 문제있는 부분 추출

    # 트랜스포터 사용 시각 추가
    '''이거는 아직 구현 안했는데.... 만약 같은 시간에 진입한다면 속도 더 빠른 놈을 앞에 배치!! 느린 놈은 앞서 간 놈의 m/s가 자기 길이 만큼 가고 나서 가는 걸로
    지연 시간은 각 트랜스포터 클래스에서 미리 만들어 놔야 할 듯'''
    def add_(self, trans, data):
        start_time = data[0]
        end_time = data[1]
        start_node = data[2]
        end_node = data[3]

        if start_node == self.s.no:
            if trans is not self.in_s_e:
                self.in_s_e[start_time] = [trans]
                self.out_s_e[end_time] = [trans]
            else:
                self.in_s_e[start_time] = self.in_s_e[start_time].append(trans)
                self.out_s_e[end_time] = self.in_s_e[end_time].append(trans)
        else:
            if trans is not self.in_e_s:
                self.in_e_s[start_time] = [trans]
                self.out_e_s[end_time] = [trans]
            else:
                self.in_e_s[start_time] = self.in_e_s[start_time].append(trans)
                self.out_e_s[end_time] = self.out_e_s[end_time].append(trans)

    # 시간 순 정렬
    def sort_dict(self):
        if len(self.s_e) > 0:
            self.in_s_e = sorted(self.in_s_e.items(), key=lambda item: item[0])
            self.out_s_e = sorted(self.out_s_e.items(), key=lambda item: item[0])
            self.in_s_e = dict(self.in_s_e)
            self.out_s_e = dict(self.out_s_e)

        if len(self.e_s) > 0:
            self.in_e_s = sorted(self.in_e_s.items(), key=lambda item: item[0])
            self.out_e_s = sorted(self.out_e_s.items(), key=lambda item: item[0])
            self.in_e_s = dict(self.in_e_s)
            self.out_e_s = dict(self.out_e_s)

        self.conflict()
            
    # 충돌 검사 --> 가장 빠른 시간대의 문제만..
    def conflict(self):
        pair =[]        # 문제되는 트랜스포터 쌍
        out_s_ek = list(self.out_s_e.keys())
        out_s_ev = list(self.out_s_e.values())
        out_e_sk = list(self.out_e_s.keys())
        out_e_sv = list(self.out_e_s.values())

        # 양 방향에서 오는데 문제되는거
        for i, k, v in enumerate(self.in_s_e.items()):
            end_time = 0
            for k2, v2 in zip(out_s_ek[i:], out_s_ev[i:]):
                if v == v2:
                    end_time = k2

            # 다른 방향과 비교
            for i2, k2, v2 in enumerate(self.in_e_s.items()):
                for k3, v3 in zip(out_e_sk[i2:], out_e_sv[i2:]):
                    if v2 == v3:
                        # 겹치지 않는다면
                        if not (k < k2 < end_time or k2 < k < k3):
                            # 다른 쪽의 시작 시간이 기준의 시작 시간보다 크다면 이후에꺼는 볼 필요 X
                            if k < k2:
                                pass_flag = True
                        # 겹친다면
                        if v.size + v2.size <= self.w:
                            continue

                        # 문제되는거임..
                        else:
                            if k < k2:
                                if k3 < end_time:
                                    data = (v, v2, k, end_time)
                                else:
                                    data = (v, v2, k, k3)
                            else:
                                if k3 < end_time:
                                    data = (v, v2, k2, end_time)
                                else:
                                    data = (v, v2, k2, k3)
                            pair = self.pair_check(pair, data)
                            
        # 시작점 --> 끝점 이동 중 오류
        for i, k, v in enumerate(self.in_s_e):
            # 같은 방향에 있는 트랜스포터 검사
            for k2, v2 in zip(out_s_ek[i:], out_s_ev[i:]):
                if v == v2:
                    break
                else:
                    data = (v, v2, k, k2)
                    pair = self.pair_check(pair, data)


        # 끝점 --> 시작점 이동 중 오류
        for i, k, v in enumerate(self.in_e_s):
            # 같은 방향에 있는 트랜스포터 검사
            for k2, v2 in zip(out_e_sk[i:], out_e_sv[i:]):
                # 가장 빠른 문제 되는 점보다 이후의 시각이면 stop!!
                if v == v2:
                    break
                else:
                    data = (v, v2, k, k2)
                    pair = self.pair_check(pair, data)

        trans, timez = self.conflict_trans(pair)

        return trans, timez

    # 문제되는 거 시간 순 비교
    def pair_check(self, pair, data):
        result = pair.copy()
        pre_flag = False
        for p in pair:
            # 앞선 일임
            if p[2] > data[3]:
                pre_flag = True
            
            # 나중에 일어난 일
            elif p[3] < data[2]:
                pre_flag = False

            else:
                # 시간 겹치는 거
                if data[0] in p or data[1] in p:
                    # 트랜스포터 연관되어 있으면 추가!!
                    result.append(data)
                    pre_flag = False
                    break

        if pre_flag:
            result = [data]

        return result

    # 가장 많이 충돌되는 트랜스포터 추출
    def conflict_trans(self, pair):
        c_dict = {}
        can = []
        min_time = pair[0][2]
        for p in pair:
            # 더 빠르다면
            if min_time >= p[2]:
                can = [p[0], p[1], p[2]]
                min_time = p[2]
            
            # 딕셔너리 추가
            if p[0] not in c_dict:
                c_dict[p[0]] = 1
            else:
                c_dict[p[0]] = c_dict[p[0]] + 1

        if c_dict[can[0]] < c_dict[can[1]]:
            min_time = can[2]
            can = can[1]
        else:
            min_time = can[2]
            can = can[0]

        return can, min_time

    # 현재 도로 위의 총 트랜스포터 대수
    def total_trans(self):
        return len(self.s_e) + len(self.e_s)
