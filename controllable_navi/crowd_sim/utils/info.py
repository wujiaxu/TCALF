class InfoList(object):
    def __init__(self):
        self.info_list = []
    def add(self,info):
        self.info_list.append(str(info))
    def contain(self,info):
        return str(info) in self.info_list
    def reset(self):
        self.info_list.clear()
    def empty(self):
        return len(self.info_list)==0

class Timeout(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Timeout'
    
class ViolateSpeedLimit(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'violating speed limit'

class ReachGoal(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Reaching goal'

class ViolateSidePreference(object):
    def __init__(self,wrong_side=None):
        self.wrong_side = wrong_side
    def __str__(self):
        return 'violating side preference'
    
class Discomfort(object):
    def __init__(self, min_dist ):
        self.min_dist = min_dist
        self.num = 0

    def __str__(self):
        return 'Discomfort'


class Collision(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Collision'


class Nothing(object):
    def __init__(self):
        pass

    def __str__(self):
        return ''
