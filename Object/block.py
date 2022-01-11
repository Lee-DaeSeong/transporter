class Block:
    def __init__(self, no, size_x, size_y, weight):
        self.no = no
        self.s_x = size_x
        self.s_y = size_y
        self.w = weight

    # 블록 최대 크기
    def b_s(self):
        if self.s_x < self.s_y:
            return self.s_y
        return self.s_x
