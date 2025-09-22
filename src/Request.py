class Request(object):
    """
    A Request is the atomic unit in an Action.

    It represents a single customer's *request* for a ride
    """

    MAX_PICKUP_DELAY: float = 100.0  # Reduced from 300.0 to match episode length
    MAX_DROPOFF_DELAY: float = 600.0

    def __init__(self,
                 request_id: int,
                 source: int,
                 destination: int,
                 current_time: float,
                 travel_time: float,
                 value: float=10,
                 final_value: float=10
                 ):
        self.request_id = request_id
        self.pickup = source
        self.dropoff = destination
        self.value = value  # In the deafult case, all requests have equal value
        self.pickup_deadline = current_time + self.MAX_PICKUP_DELAY
        self.dropoff_deadline = current_time + travel_time + self.MAX_DROPOFF_DELAY
        self.final_value = final_value  # 修正：应该使用final_value参数而不是value
    def __deepcopy__(self, memo):
        return self

    def __str__(self):
        return("{}->{}".format(self.pickup, self.dropoff))

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.request_id)

    def __eq__(self, other):
        # Request is only comparable with other Requests
        if isinstance(other, self.__class__):
            # If the ids are the same, they are equal
            if (self.request_id == other.request_id):
                return True
        
        return False
