class ValidationPoints():
    def __init__(self):
        #down-up
        self.down_up = [
            [[11, 11], [13, 75]],
            [[16, 16], [18, 70]],
            [[26, 26], [28, 85]],
            [[31, 21], [33, 90]],
            [[41, 26], [43, 95]],
            [[56, 31], [58, 80]],
            [[61, 36], [63, 85]],
            [[71, 41], [73, 90]]
        ]

        #left-right
        self.left_right = [
            [[21, 51], [75, 53]],
            [[31, 41], [85, 43]],
            [[46, 56], [90, 58]],
            [[51, 61], [95, 63]],
            [[26, 46], [80, 48]],
            [[36, 66], [90, 68]],
            [[61, 81], [75, 83]],
            [[71, 71], [85, 73]]
        ]

        self.up_down = [
            [[24, 69], [22, 5]],
            [[33, 79], [31, 15]],
            [[44, 74], [42, 10]],
            [[51, 84], [53, 20]],
            [[69, 89], [67, 25]],
            [[74, 94], [72, 30]],
            [[81, 69], [83, 5]]]

        self.right_left = [
            [[79, 44], [15, 42]],
            [[74, 34], [10, 32]],
            [[84, 54], [20, 52]],
            [[89, 22], [25, 24]],
            [[94, 68], [30, 66]],
            [[69, 76], [5, 78]],
            [[79, 84], [15, 82]]
        ]

        self.ld_ru = [
            [[11, 16], [46, 80]],
            [[16, 21], [51, 85]],
            [[26, 31], [61, 75]],
            [[31, 36], [66, 80]],
            [[41, 46], [76, 85]],
            [[51, 56], [86, 75]]
        ]
        self.ru_ld = [
            [[52, 76],[21, 20]],
            [[61, 82],[32, 25]],
            [[57, 89], [41, 25]],
            [[46,60], [9,10]],
            [[79, 68], [52, 10]],
            [[71, 64], [16, 30]]
        ]
    def get_down_up(self):
        return self.down_up

    def get_left_right(self):
        return self.left_right

    def get_up_down(self):
        return self.up_down

    def get_right_left(self):
        return self.right_left

    def get_ld_ru(self):
        return self.ld_ru