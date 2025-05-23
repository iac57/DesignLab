class RepeaterCheck:
    def __init__(self):
        self.count = 0
        self.prev_machine = 10

    def check(self, machine):
        if machine == self.prev_machine:
            self.count += 1
            if self.count > 2:
                return 1
        else:
            self.count =0
            return 0
    
    def update(self, machine):
        self.prev_machine = machine