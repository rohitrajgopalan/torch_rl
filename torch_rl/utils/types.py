import enum

class NetworkOptimizer(enum.Enum):
    ADAM = 1,
    ADAMAX = 2,
    RMSPROP = 3,
    SGD = 4,
    ADAGRAD = 5,

    @staticmethod
    def all():
        return [NetworkOptimizer.ADAM, NetworkOptimizer.ADAMAX, NetworkOptimizer.RMSPROP, NetworkOptimizer.SGD,
                NetworkOptimizer.ADAGRAD]

    @staticmethod
    def get_type_by_name(name):
        for optimizer_type in NetworkOptimizer.all():
            if optimizer_type.name.lower() == name.lower():
                return optimizer_type
        return None
