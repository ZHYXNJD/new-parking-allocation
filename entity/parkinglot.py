class Parkinglot:
    def __init__(self, id, total_num, charge_num, slow_charge_num, park_fee=3, charge_fee=1.5,
                 reserve_fee=1):  # 初始化一个停车场，需要定义停车场编号，泊位数量，充电桩比例
        self.id = id
        self.total_num = total_num
        self.charge_num = charge_num
        self.ordinary_num = total_num - charge_num
        self.slow_charge_space = slow_charge_num
        self.fast_charge_space = charge_num - slow_charge_num
        self.park_fee = park_fee
        self.charge_fee = charge_fee
        self.reserve_fee = reserve_fee


def get_parking_lot(parking_lot_num=4):
    # 4个停车场
    pl1 = Parkinglot(id=1, total_num=40, charge_num=10, slow_charge_num=3)
    pl2 = Parkinglot(id=2, total_num=20, charge_num=5, slow_charge_num=1)
    pl3 = Parkinglot(id=3, total_num=10, charge_num=2, slow_charge_num=1)
    pl4 = Parkinglot(id=4, total_num=30, charge_num=8, slow_charge_num=2)
    pl = [pl1, pl2, pl3, pl4]
    current_pl_num = len(pl)
    if current_pl_num < parking_lot_num:
        print(f"目前已初始化{current_pl_num}个停车场，还需要新添加{parking_lot_num - current_pl_num}个停车场")
    elif current_pl_num > parking_lot_num:
        return pl[:parking_lot_num]
    else:
        return pl
