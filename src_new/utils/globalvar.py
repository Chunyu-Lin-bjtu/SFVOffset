class GlobalVar:
    """global var for conv counter"""
    def __init__(self, num=0):
        self.conv_counter = 0
    
    def add1_conv_counter(self):
        self.conv_counter = self.conv_counter + 1
    def get_conv_counter(self):
        return self.conv_counter