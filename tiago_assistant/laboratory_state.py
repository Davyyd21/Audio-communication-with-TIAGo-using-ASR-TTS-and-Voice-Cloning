class LaboratoryState:

    def __init__(self):
        self.active_laboratory = None

    def set_active_laboratory(self,laboratory_name: str,)->None:
        self.active_laboratory = laboratory_name

    def get_active_laboratory(self):
        return self.active_laboratory

    def clear(self) -> None:
        self.active_laboratory = None
