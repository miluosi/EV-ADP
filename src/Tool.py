class Tool:
    def __init__(self, name, version):
        self.name = name
        self.version = version

    def get_info(self):
        return f"{self.name} (version {self.version})"
    


    