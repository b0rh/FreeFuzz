import json
import inspect
from numpy.random import choice, randint
import torch
from enum import Enum
import random
from os.path import join
import simAPI

example_record = '{"parameter:0": 16, "parameter:1": 16, "parameter:2": [1, 3], "stride": 1, "padding": [0, 1], "bias": true, "dilation": [1, 1], "input_signature": [{"shape": [8, 3, 256, 512], "dtype": "torch.float32"}], "output_signature": {"shape": [8, 13, 128, 256], "dtype": "torch.float32"}}'


class ArgType(Enum):
    INT = 1
    STR = 2
    FLOAT = 3
    BOOL = 4
    TENSOR_OBJECT = 5
    TUPLE = 6
    LIST = 7
    TENSOR = 8
    NULL = 9
    TORCH_DTYPE = 10


class Argument:
    _int_values = [-16, -1, 0, 1, 16]
    _str_values = [
        "mean", "sum", "max", 'zeros', 'reflect', 'circular', 'replicate'
    ]
    _float_values = [0.0, 1.0, -1.0, 63.0, -63.0]
    _dtypes = [
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
        torch.float16, torch.float32, torch.bfloat16, torch.complex32,
        torch.complex64, torch.complex128
    ]

    def __init__(self, value, type: ArgType, max_value=0, min_value=0) -> None:
        self.value = value
        self.type = type
        # the max value for tensor.randint
        self.max_value = max_value
        self.min_value = min_value

    def get_value(self):
        if self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            # self.value is a list now
            temp_value = []
            for arg in self.value:
                temp_value.append(arg.get_value())
            if self.type == ArgType.TUPLE:
                return tuple(temp_value)
            else:
                return temp_value
        else:
            return self.value

    def to_code(self, var_name) -> str:
        if self.type in [ArgType.INT, ArgType.FLOAT, ArgType.BOOL]:
            return "  %s = %s\n" % (var_name, self.value)
        elif self.type == ArgType.STR:
            return "  %s = \"%s\"\n" % (var_name, self.value)
        elif self.type == ArgType.LIST:
            code = ""
            for i in range(len(self.value)):
                code += self.value[i].to_code("%s_%s" % (var_name, i))
            code += "  %s = [" % (var_name)
            for i in range(len(self.value)):
                code += "%s_%s," % (var_name, i)
            code += "]\n"
            return code
        elif self.type == ArgType.TUPLE:
            code = ""
            for i in range(len(self.value)):
                code += self.value[i].to_code("%s_%s" % (var_name, i))
            code += "  %s = (" % (var_name)
            for i in range(len(self.value)):
                code += "%s_%s," % (var_name, i)
            code += ")\n"
            return code
        elif self.type == ArgType.NULL:
            return "  %s = None\n" % (var_name)
        elif self.type == ArgType.TENSOR:
            dtype = self.value.dtype
            if dtype.is_floating_point:
                return "  %s = torch.rand(%s, dtype=%s)\n" % (
                    var_name, self.value.shape, dtype)
            elif dtype.is_complex:
                return "  %s = torch.rand(%s, dtype=%s)\n" % (
                    var_name, self.value.shape, dtype)
            elif dtype == torch.bool:
                return "  %s = torch.randint(0,2,%s, dtype=%s)\n" % (
                    var_name, self.value.shape, dtype)
            else:
                return "  %s = torch.randint(%s,%s,%s, dtype=%s)\n" % (
                    var_name, self.min_value, self.max_value, self.value.shape,
                    dtype)
        elif self.type == ArgType.TENSOR_OBJECT:
            if isinstance(self.value, torch.device):
                return "  %s = torch.device(\"cpu\")\n" % (var_name)
            return "  %s = %s\n" % (var_name, self.value)
        elif self.type == ArgType.TORCH_DTYPE:
            return "  %s = %s\n" % (var_name, self.value)
        else:
            assert (0)

    def mutate(self, arg_name, APIs=None, probs=None) -> None:
        def doTypeMutation():
            return randint(0, 3) == 0

        def selectRandOverDb():
            return randint(0, 3) == 0

        if doTypeMutation():
            self.mutate_type()
        doValueMutation = True
        if selectRandOverDb():
            if APIs != None and len(APIs):
                target_api = choice(APIs, p=probs)
                new_arg = Argument.select_rand_over_db(target_api, arg_name)
                if new_arg:
                    self.value = new_arg.value
                    self.type = new_arg.type
                    self.max_value = new_arg.max_value
                    self.min_value = new_arg.min_value
                    doValueMutation = False
        if doValueMutation:
            self.mutate_value()

    def mutate_value(self) -> None:
        if self.type == ArgType.INT:
            self.value = Argument.mutate_int_value(self.value)
        elif self.type == ArgType.STR:
            self.value = Argument.mutate_str_value(self.value)
        elif self.type == ArgType.FLOAT:
            self.value = Argument.mutate_float_value(self.value)
        elif self.type == ArgType.BOOL:
            # change the bool value
            self.value = not self.value
        elif self.type == ArgType.TENSOR_OBJECT:
            pass
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            # self.value is a list now
            for arg in self.value:
                arg.mutate_value()
        elif self.type == ArgType.TENSOR:
            self.value, self.max_value, self.min_value = Argument.mutate_tensor_value(
                self.value)
        elif self.type == ArgType.NULL:
            pass
        elif self.type == ArgType.TORCH_DTYPE:
            # same with type mutation for dtype
            self.value = choice(Argument._dtypes)
        else:
            assert (0)

    def mutate_type(self) -> None:
        if self.type in [
                ArgType.INT, ArgType.FLOAT, ArgType.STR, ArgType.BOOL
        ]:
            # change the type
            types = [ArgType.INT, ArgType.FLOAT, ArgType.STR, ArgType.BOOL]
            types.remove(self.type)
            self.type = choice(types)
            # change the value
            if self.type == ArgType.INT:
                self.value = Argument.mutate_int_value(0)
            elif self.type == ArgType.FLOAT:
                self.value = Argument.mutate_float_value(0.0)
            elif self.type == ArgType.STR:
                self.value = Argument.mutate_str_value("max")
            elif self.type == ArgType.BOOL:
                self.value = choice([True, False])
        elif self.type in [ArgType.LIST, ArgType.TUPLE]:
            for arg in self.value:
                arg.mutate_type()
        elif self.type == ArgType.TENSOR:
            dtype = choice(Argument._dtypes)
            new_size = list(self.value.shape)
            if randint(0, 3) == 0:
                # change the dimension
                if randint(0, 3) == 0:
                    if randint(0, 1):
                        new_size.append(1)
                    elif len(new_size) > 0:
                        new_size.pop()
                # change the size value
                for i in range(len(new_size)):
                    if randint(0, 2) == 0:
                        new_size[i] = Argument.mutate_int_value(new_size[i],
                                                                _min=1)

            self.value, self.max_value, self.min_value = Argument.random_tensor_value(
                new_size, dtype)
        elif self.type == ArgType.TENSOR_OBJECT:
            pass
        elif self.type == ArgType.NULL:
            new_type = choice(ArgType)
            if new_type == ArgType.LIST or new_type == ArgType.TUPLE:
                self.value = [
                    Argument(2, ArgType.INT),
                    Argument(3, ArgType.INT)
                ]
            elif new_type == ArgType.TENSOR:
                self.value = torch.rand([2, 2])

            if new_type != ArgType.NULL:
                self.type = new_type
                self.mutate_type()
        elif self.type == ArgType.TORCH_DTYPE:
            self.value = choice(Argument._dtypes)
        else:
            assert (0)

    @staticmethod
    def mutate_int_value(value, _min=None, _max=None) -> int:
        if randint(0, 10) == 0:
            value = choice(Argument._int_values)
        else:
            value += randint(-16, 16)
        # min <= value <= max
        if _min:
            value = max(_min, value)
        if _max:
            value = min(_max, value)
        return value

    @staticmethod
    def mutate_str_value(value) -> str:
        if randint(0, 10) == 0:
            return choice(Argument._str_values)
        else:
            return value

    @staticmethod
    def mutate_float_value(value) -> float:
        if randint(0, 10) == 0:
            return choice(Argument._float_values)
        else:
            return value + randint(-16, 16)

    @staticmethod
    def mutate_tensor_value(value):
        return Argument.random_tensor_value(value.shape, value.dtype)

    @staticmethod
    def random_tensor_value(size, dtype):
        try:
            size = list(size)
            for i in range(len(size)):
                size[i] = max(1, size[i])
            res = None
            max_value = 0
            min_value = 0
            if dtype.is_floating_point:
                res = torch.rand(size, dtype=dtype)
            elif dtype.is_complex:
                res = torch.rand(size, dtype=dtype)
            elif dtype == torch.bool:
                res = torch.randint(0, 2, size, dtype=dtype)
            elif dtype == torch.int8:
                max_value = 1 << 7 if randint(0, 2) else 1
                min_value = -1 << 7 if randint(0, 2) else 0
                res = torch.randint(min_value, max_value, size, dtype=dtype)
            elif dtype == torch.int16:
                max_value = 1 << 15 if randint(0, 2) else 1
                min_value = -1 << 15 if randint(0, 2) else 0
                res = torch.randint(min_value, max_value, size, dtype=dtype)
            elif dtype == torch.uint8:
                max_value = 1 << 8 if randint(0, 2) else 1
                min_value = 0
                res = torch.randint(min_value, max_value, size, dtype=dtype)
            else:
                max_value = 1 << 15 if randint(0, 2) else 1
                min_value = -1 << 15 if randint(0, 2) else 0
                res = torch.randint(min_value, max_value, size, dtype=dtype)
            return res, max_value, min_value
        except Exception:
            # allocate error
            return torch.rand([2, 2]), 0, 0

    @staticmethod
    def select_rand_over_db(api_name, arg_name):
        if not api_name.startswith("torch."):
            print("not start with!")
            return None
        file_name = api_name[6:]
        try:
            with open("argVS/%s.txt" % (file_name), "r") as f:
                value_space = json.loads(f.read())
                if arg_name in value_space.keys():
                    signature = random.choice(value_space[arg_name])
                    return PytorchAPI.generate_arg_from_signature(signature)
                else:
                    return None
        except FileNotFoundError:
            return None


class PytorchAPI:
    def __init__(self, api_name, record, filename="", args=None) -> None:
        self.record = record
        self.api = api_name
        self.args = args if args else PytorchAPI.generate_args_from_record(
            self.record, self.api)
        self.filename = filename

    def mutate(self,
               default_args: dict = {},
               name_list: list = [],
               arg_database: dict = {}) -> None:
        if randint(0, 3) == 0 and len(default_args):
            new_default_key = choice(list(default_args.keys()))
            if new_default_key not in self.args.keys():
                self.args[new_default_key] = default_args[new_default_key]

        num_arg = len(self.args)
        if num_arg == 0:
            return
        num_Mutation = randint(1, num_arg + 1)
        for _ in range(num_Mutation):
            arg_key = choice(list(self.args.keys()))
            arg_name = arg_key
            APIs = None
            probs = None
            if arg_key.startswith("parameter:"):
                index = int(arg_key[10:])
                if index < len(name_list):
                    arg_name = name_list[index]
                    if arg_name in arg_database.keys():
                        APIs, probs = arg_database[arg_name]
            elif arg_key in arg_database.keys():
                APIs, probs = arg_database[arg_key]
            self.args[arg_key].mutate(arg_name, APIs, probs)

    def to_code(self) -> str:
        args = []
        kwargs = {}
        inputs = None
        for key in self.args.keys():
            if "parameter:" in key:
                args.append(self.args[key])
            elif key == "input_signature":
                inputs = self.args[key]
            elif key != "output_signature":
                kwargs[key] = self.args[key]
        code = ""
        arg_str = ""
        index = 0
        func = eval(self.api)
        for arg in args:
            code += arg.to_code("arg_%s" % (index))
            arg_str += "arg_%s," % (index)
            index += 1
        for key, arg in kwargs.items():
            code += arg.to_code(key)
            arg_str += "%s=%s," % (key, key)
        if inspect.isclass(func):
            code += "  res = %s(%s)\n" % (self.api, arg_str)
            if inputs:
                code += inputs.to_code("ins")
                code += "  res(*ins)\n"
        else:
            code += "  res = %s(%s)\n" % (self.api, arg_str)
        return code

    @staticmethod
    def generate_arg_from_signature(signature):
        # signature is a simple object
        if signature == "torchTensor":
            return Argument(torch.randint(0, 2, [2, 3]), ArgType.TENSOR)
        if signature == "torchdtype":
            return Argument(choice(Argument._dtypes), ArgType.TORCH_DTYPE)
        if isinstance(signature, str) and signature == "torchdevice":
            value = torch.device(randint(0, 5))
            return Argument(value, ArgType.TENSOR_OBJECT)
        if isinstance(signature, str) and signature == "torchmemory_format":
            value = choice([
                torch.contiguous_format, torch.channels_last,
                torch.preserve_format
            ])
            return Argument(value, ArgType.TENSOR_OBJECT)
        if isinstance(signature, bool):
            return Argument(signature, ArgType.BOOL)
        if isinstance(signature, int):
            return Argument(signature, ArgType.INT)
        if isinstance(signature, str):
            return Argument(signature, ArgType.STR)
        if isinstance(signature, float):
            return Argument(signature, ArgType.FLOAT)
        if isinstance(signature, tuple):
            value = []
            for elem in signature:
                value.append(PytorchAPI.generate_arg_from_signature(elem))
            return Argument(value, ArgType.TUPLE)
        if isinstance(signature, list):
            value = []
            for elem in signature:
                value.append(PytorchAPI.generate_arg_from_signature(elem))
            return Argument(value, ArgType.LIST)
        # signature is a dictionary
        if isinstance(signature, dict):
            if not ('shape' in signature.keys()
                    and 'dtype' in signature.keys()):
                raise Exception('Wrong signature {0}'.format(signature))
            shape = signature['shape']
            dtype = signature['dtype']
            # signature is a ndarray or tensor.
            if isinstance(shape, (list, tuple)):
                if not dtype.startswith("torch."):
                    dtype = "torch.%s" % (dtype)
                dtype = eval(dtype)
                value, max_value, min_value = Argument.random_tensor_value(
                    shape, dtype)
                return Argument(value, ArgType.TENSOR, max_value, min_value)
            else:
                temp_tensor = torch.tensor([2, 2])
                return Argument(temp_tensor, ArgType.TENSOR)
        return Argument(None, ArgType.NULL)

    @staticmethod
    def generate_args_from_record(record: dict, name: str = ""):

        args = {}
        for key in record.keys():
            if key != "output_signature":
                args[key] = PytorchAPI.generate_arg_from_signature(record[key])
        return args


def parse_line(line: str):
    res = {}
    name_list = []
    args = line.split(", ")
    for arg in args:
        if "=" in arg:
            # default value
            temp = arg.split("=")
            assert (len(temp) == 2)
            key = temp[0]
            value = temp[1]

            if key == "dtype":
                res[key] = Argument(eval(value), ArgType.TORCH_DTYPE)
            elif "pickle" in key:
                continue
            elif value.startswith("torch."):
                res[key] = Argument(eval(value), ArgType.TENSOR_OBJECT)
            else:
                value = eval(value)
                if isinstance(value, bool):
                    res[key] = Argument(value, ArgType.BOOL)
                elif isinstance(value, int):
                    res[key] = Argument(value, ArgType.INT)
                elif isinstance(value, float):
                    res[key] = Argument(value, ArgType.FLOAT)
                elif isinstance(value, str):
                    res[key] = Argument(value, ArgType.STR)
                elif value == None:
                    res[key] = Argument(value, ArgType.NULL)
                else:
                    assert (0)
        else:
            name_list.append(arg)
    return res, name_list


def output(filename, code):
    with open(join("..", "output", filename), "w") as f:
        f.write("import torch\n")
        f.write("try:\n")
        f.write(code)
        f.write("except Exception:\n  pass\n")


example_api_name = "torch.nn.Conv2d"

default_args, name_list = parse_line(simAPI.API_args[example_api_name])

example_record = json.loads(example_record)
Conv2d = PytorchAPI(example_api_name, record=example_record)
output("example_conv2d.py", Conv2d.to_code())

Conv2d.mutate(default_args, name_list)
output("example_conv2d_mutation.py", Conv2d.to_code())