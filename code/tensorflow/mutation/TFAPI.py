import inspect
import json
import os
from enum import Enum

import random
from numpy.random import choice, randint
import tensorflow as tf
from termcolor import colored

from mutation.arg_vs import ArgmentValueSpace
from mutation.random_utils import *
from mutation.read_db import get_rand_single_signature
from mutation.db_mutation import find_sim_api_arg


class ArgType(Enum):
    INT = 1
    STR = 2
    FLOAT = 3
    BOOL = 4
    TUPLE = 5
    LIST = 6
    TENSOR = 7
    TF_DTYPE = 8
    KERAS_TENSOR = 9
    VARIABLE = 10
    TF_OBJECT = 11
    NULL = 12


class Argument:
    _int_values = [-16, -1, 0, 1, 16]
    _str_values = ["", "1", "sum", "same", "valid", "zeros"]
    _float_values = [0.0, 1.0, -1.0, 63.0, -63.0]
    _tensor_arg_dtypes = [ArgType.TENSOR, ArgType.KERAS_TENSOR, ArgType.VARIABLE]
    _dtypes = [
        tf.bfloat16, tf.bool, tf.complex128, tf.complex64, tf.double,
        tf.float16, tf.float32, tf.float64, tf.half,
        tf.int16, tf.int32, tf.int64, tf.int8,
        tf.uint8, tf.uint16, tf.uint32, tf.uint64,
    ]
    _arg_vs = ArgmentValueSpace()

    def __init__(self, value, type: ArgType, minv=0, maxv=0, shape=None, dtype=None) -> None:
        self.value = value
        self.type = type
        self.minv = minv
        self.maxv = maxv
        self.shape = shape
        self.dtype = dtype
        if isinstance(dtype, str):
            self.dtype = self.str_to_dtype(dtype)

    @staticmethod
    def str_to_dtype(dt: str):
        dt = dt.strip().replace("_ref", "")
        if not dt.startswith("tf."):
            dt = "tf." + dt
        try:
            return eval(dt)
        except:
            return tf.float32

    def __copy__(self):
        if self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            temp_value = []
            for arg in self.value:
                temp_value.append(arg.__copy__())
            return Argument(temp_value, self.type, self.minv,
                            self.maxv, self.shape, self.dtype)
        else:
            return Argument(self.value, self.type, self.minv, 
                            self.maxv, self.shape, self.dtype)

    def get_value(self):
        """ Returns the value of the current api. """
        if self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            temp_value = []
            for arg in self.value:
                temp_value.append(arg.get_value())
            return temp_value
        elif self.type == ArgType.TENSOR:
            return random_tensor(self.shape, self.dtype)
        elif self.type == ArgType.KERAS_TENSOR:
            return random_keras_tensor(self.shape, self.dtype)
        elif self.type == ArgType.VARIABLE:
            return random_variable(self.shape, self.dtype)
        else:
            return self.value

    def mutate_value_random(self) -> None:
        """ Apply random value mutation. """
        if self.type == ArgType.INT:
            self.value = Argument.mutate_int_value(self.value)
        elif self.type == ArgType.STR:
            self.value = Argument.mutate_str_value(self.value)
        elif self.type == ArgType.FLOAT:
            self.value = Argument.mutate_float_value(self.value)
        elif self.type == ArgType.BOOL:
            self.value = Argument.mutate_bool_value(self.value)
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            for arg in self.value:
                arg.mutate_value_random()
        elif self.type in self._tensor_arg_dtypes:
            pass
        elif self.type == ArgType.TF_DTYPE:
            self.value = Argument.mutate_dtype()
        elif self.type == ArgType.TF_OBJECT:
            pass
        elif self.type == ArgType.NULL:
            pass
        else:
            assert (0)

    def mutate_value_db(self, api_name, arg_name):

        if self.type == ArgType.NULL:
            return False

        if self.type in [ArgType.TENSOR, ArgType.KERAS_TENSOR, ArgType.VARIABLE]:
            return False

        mut_api = find_sim_api_arg(api_name, arg_name)
        if mut_api is None:
            return False

        mut_sig = self._arg_vs.sample_arg_signature_from_vs(mut_api, arg_name)
        if mut_sig is None:
            return False
        mut_arg = generate_arg_from_signature(mut_sig)
        self.type = mut_arg.type
        self.value = mut_arg.value
        self.minv = mut_arg.minv
        self.maxv = mut_arg.maxv
        self.dtype = mut_arg.dtype
        self.shape = mut_arg.shape
        return True

    def if_mutate_shape(self):
        return random.random() < 0.3

    def if_mutate_shape_value(self):
        return random.random() < 0.3

    def if_expand_dim(self):
        return random.random() < 0.3

    def if_squeeze(self):
        return random.random() < 0.3

    def mutate_shape(self, old_shape):
        new_shape = old_shape
        if not isinstance(new_shape, list):
            try:
                new_shape = new_shape.as_list()
            except:
                new_shape = list(new_shape)
            else:
                new_shape = list(new_shape)
        for i in new_shape:
            if not np.issubdtype(type(i), np.integer):
                assert (0)

        # Change rank
        if self.if_expand_dim():
            new_shape.append(1)
        elif len(new_shape) > 0 and self.if_squeeze():
            new_shape.pop()
        # Change value
        for i in range(len(new_shape)):
            if self.if_mutate_shape_value():
                new_shape[i] = Argument.mutate_int_value(new_shape[i], minv=0)
        return new_shape

    def generate_value_random(self) -> None:

        if self.type == ArgType.INT:
            self.value = Argument.mutate_int_value(0)
        elif self.type == ArgType.STR:
            self.value = Argument.mutate_str_value("")
        elif self.type == ArgType.FLOAT:
            self.value = Argument.mutate_float_value(0.)
        elif self.type == ArgType.BOOL:
            self.value = Argument.mutate_bool_value(True)
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            self.value = [Argument(1, ArgType.INT), Argument(1, ArgType.INT)]
        elif self.type in self._tensor_arg_dtypes:
            shape = [randint(1,3), randint(1,3)]
            dtype = choice([tf.int32, tf.float32, tf.float64])
            self.shape, self.dtype = shape, dtype
            self.value, self.minv, self.maxv = None, 0, 1
        elif self.type == ArgType.TF_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.TF_OBJECT:
            self.value = None
            pass
        elif self.type == ArgType.NULL:
            self.value = None
            pass
        else:
            assert (0)


    def mutate_type(self) -> None:
        def if_mutate_primitive():
            return random.random() < 0.1
        def if_mutate_null():
            return random.random() < 0.1
        if self.type in [ArgType.INT, ArgType.FLOAT, ArgType.STR, ArgType.BOOL]:
            if not if_mutate_primitive(): return False
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
                self.value = Argument.mutate_str_value("")
            elif self.type == ArgType.BOOL:
                self.value = choice([True, False])
        elif self.type in [ArgType.LIST, ArgType.TUPLE]:
            for arg in self.value:
                arg.mutate_type()
        elif self.type == ArgType.TENSOR:
            dtype = choice(Argument._dtypes)
            shape = self.shape
            if self.if_mutate_shape():
                shape = self.mutate_shape(shape)
            self.shape, self.dtype = shape, dtype
        elif self.type == ArgType.TF_OBJECT:
            pass
        elif self.type == ArgType.NULL:
            if not if_mutate_null():
                return False
            new_type = choice(ArgType)
            if new_type == ArgType.LIST or new_type == ArgType.TUPLE:
                self.value = [
                    Argument(2, ArgType.INT),
                    Argument(3, ArgType.INT)
                ]
            elif new_type == ArgType.TENSOR:
                self.shape = [2, 2]
                self.dtype = tf.float32

            if new_type != ArgType.NULL:
                try:
                    self.type = new_type
                    self.generate_value_random()
                except:
                    pass
        elif self.type == ArgType.TF_DTYPE:
            self.value = choice(Argument._dtypes)

    @staticmethod
    def if_mutate_int_random():
        return random.random() < 0.3
    @staticmethod
    def if_mutate_str_random():
        return random.random() < 0.1
    @staticmethod
    def if_mutate_float_random():
        return random.random() < 0.3
        
    @staticmethod
    def mutate_int_value(value, minv=None, maxv=None) -> int:
        
        if Argument.if_mutate_int_random():
            value = choice(Argument._int_values)
        else:
            value += randint(-2, 2)
        if minv:
            value = max(minv, value)
        if maxv:
            value = min(maxv, value)
        return value

    @staticmethod
    def mutate_str_value(value) -> str:
        if Argument.if_mutate_str_random():
            return choice(Argument._str_values)
        return value

    @staticmethod
    def mutate_float_value(value) -> float:
        if Argument.if_mutate_float_random():
            return choice(Argument._float_values)
        else:
            return value + randint(-16, 16)

    @staticmethod
    def mutate_bool_value(value) -> bool:
        return choice([True, False])

    @staticmethod
    def mutate_tensor_value(value) -> tf.Tensor:
        dtype = value.dtype
        shape = value.shape
        res = random_tensor(shape, dtype)
        return res

    @staticmethod
    def mutate_keras_tensor_value(value):
        res = None
        dtype = value.dtype
        shape = value.shape
        res = random_keras_tensor(shape, dtype)
        return res

    @staticmethod
    def mutate_variable_value(value) -> tf.Variable:
        res = None
        dtype = value.dtype
        shape = value.shape
        res = random_variable(shape, dtype)
        return res

    @staticmethod
    def mutate_dtype() -> tf.dtypes.DType:
        return choice(Argument._dtypes)

    def to_code_tensor(self, var_name):
        dtype = self.dtype
        shape = self.shape
        if dtype is None:
            assert(0)
        if dtype.is_floating:
            return "%s = tf.random.uniform(%s, dtype=tf.%s)\n" % (var_name, shape, dtype.name)
        elif dtype.is_complex:
            ftype = "float64" if dtype == tf.complex128 else "float32"
            return "%s = tf.complex(tf.random.uniform(%s, dtype=tf.%s)," \
                   "tf.random.uniform(%s, dtype=tf.%s))\n" % (var_name, shape, ftype, shape, ftype)
        elif dtype == tf.bool: 
            return "%s = tf.cast(tf.random.uniform(" \
                "%s, minval=0, maxval=2, dtype=tf.int32), dtype=tf.bool)\n" % (var_name, shape)
        elif dtype == tf.string:
            return "%s = tf.convert_to_tensor(np.ones(%s, dtype=str))\n" % (var_name, shape)
        else:
            return "%s = tf.saturate_cast(" \
                "tf.random.uniform(%s, minval=%d, maxval=%d, dtype=tf.int64), " \
                "dtype=tf.%s)\n" % (var_name, shape, self.minv, self.maxv + 1, dtype.name)

    def to_code_keras_tensor(self, var_name):
        return self.to_code_tensor(var_name)

    def to_code(self, var_name) -> str:
        if self.type in [ArgType.INT, ArgType.FLOAT, ArgType.BOOL]:
            if len(str(self.value)) == 0:
                self.value = 0
            return "%s = %s\n" % (var_name, str(self.value))
        elif self.type == ArgType.STR:
            return "%s = \"%s\"\n" % (var_name, self.value)
        elif self.type == ArgType.LIST:
            code = ""
            for i in range(len(self.value)):
                code += self.value[i].to_code("%s_%d" % (var_name, i))
            code += "%s = [" % (var_name)
            for i in range(len(self.value)):
                code += "%s_%d," % (var_name, i)
            code += "]\n"
            return code
        elif self.type == ArgType.TUPLE:
            code = ""
            for i in range(len(self.value)):
                code += self.value[i].to_code("%s_%d," % (var_name, i))
            code += "%s = (" % (var_name)
            for i in range(len(self.value)):
                code += "%s_%d," % (var_name, i)
            code += ")\n"
            return code
        elif self.type == ArgType.NULL:
            return "%s = None\n" % (var_name)
        elif self.type == ArgType.TENSOR:
            return self.to_code_tensor(var_name)
        elif self.type == ArgType.VARIABLE:
            return self.to_code_tensor(var_name) + "%s = tf.Variable(%s)\n" % (var_name, var_name)
        elif self.type == ArgType.KERAS_TENSOR:
            return self.to_code_keras_tensor(var_name)

        elif self.type == ArgType.TF_OBJECT:
            return "%s = None\n" % (var_name)
        elif self.type == ArgType.TF_DTYPE:
            return "%s = tf.%s\n" % (var_name, self.value.name)
        else:
            return "None"


def generate_arg_from_signature(signature):

    if isinstance(signature, int):
        return Argument(signature, ArgType.INT)
    if isinstance(signature, float):
        return Argument(signature, ArgType.FLOAT)
    if isinstance(signature, str):
        return Argument(signature, ArgType.STR)
    if isinstance(signature, bool):
        return Argument(signature, ArgType.BOOL)
    if isinstance(signature, list):
        value = []
        for elem in signature:
            value.append(generate_arg_from_signature(elem))
        return Argument(value, ArgType.LIST)
    if isinstance(signature, tuple):
        value = []
        for elem in signature:
            value.append(generate_arg_from_signature(elem))
        return Argument(value, ArgType.TUPLE)

    if (not isinstance(signature, dict)) or ('Label' not in signature):
        return Argument(None, ArgType.NULL)

    label = signature["Label"]

    if label == "tf_object":
        if signature["class_name"] == "tensorflow.python.keras.engine.keras_tensor.KerasTensor":
            dtype = signature["dtype"]
            shape = signature["shape"]
            value, minv, maxv = random_keras_tensor(shape, dtype)
            return Argument(value, ArgType.TENSOR, minv, maxv, shape, dtype)
        if signature["class_name"] == "tensorflow.python.ops.variables.RefVariable":
            dtype = signature["dtype"].replace("_ref", "")
            shape = signature["shape"]
            value, minv, maxv = random_tensor(shape, dtype)
            return Argument(value, ArgType.TENSOR, minv, maxv, shape, dtype)
        if signature["class_name"] == "tensorflow.python.framework.dtypes.DType":
            name = signature["to_str"].replace("<dtype: '", "").replace("'>", "")
            value = eval("tf." + name)
            return Argument(value, ArgType.TF_DTYPE)
        try:
            value = eval(signature.class_name)
        except:
            value = None
        return Argument(value, ArgType.TF_OBJECT)
    if label == "raw":
        try:
            value = json.loads(signature['value'])
        except:
            value = signature
            pass
        if isinstance(value, int):
            return Argument(value, ArgType.INT)
        if isinstance(value, str):
            return Argument(value, ArgType.STR)
        if isinstance(value, float):
            return Argument(value, ArgType.FLOAT)
        if isinstance(value, tuple):
            tuple_value = []
            for elem in value:
                tuple_value.append(generate_arg_from_signature(elem))
            return Argument(tuple_value, ArgType.TUPLE)
        if isinstance(value, list):
            list_value = []
            for elem in value:
                list_value.append(generate_arg_from_signature(elem))
            return Argument(list_value, ArgType.LIST)

    if label == "tuple":
        value = json.loads(signature['value'])
        tuple_value = []
        for elem in value:
            tuple_value.append(generate_arg_from_signature(elem))
        return Argument(tuple_value, ArgType.TUPLE)
    if label == "list":
        try:
            value = json.loads(signature['value'])
        except:
            value = signature['value']
        list_value = []
        for elem in value:
            list_value.append(generate_arg_from_signature(elem))
        return Argument(list_value, ArgType.LIST)
    if label == "tensor":
        if not ('shape' in signature.keys()
                and 'dtype' in signature.keys()):
            raise Exception('Wrong signature {0}'.format(signature))
        shape = signature['shape']
        dtype = signature['dtype']
        
        if isinstance(shape, (list, tuple)):
            value, minv, maxv = random_tensor(shape, dtype)
            return Argument(value, ArgType.TENSOR, minv, maxv, shape, dtype)
        else:
            minv, maxv = 0, 1
            shape = [1, ]
            return Argument(None, ArgType.TENSOR, minv, maxv, shape, dtype)
    if label == "KerasTensor":
        shape = signature['shape']
        dtype = signature['dtype']
        if isinstance(shape, (list, tuple)):
            value, minv, maxv = random_keras_tensor(shape, dtype)
            return Argument(None, ArgType.KERAS_TENSOR, minv, maxv, shape, dtype)
        else:
            raise Exception('Unknown signature {0}'.format(signature))

    if label == "variable":
        if not ('shape' in signature.keys()
                and 'dtype' in signature.keys()):
            raise Exception('Wrong signature {0}'.format(signature))
        shape = signature['shape']
        dtype = signature['dtype']
        if isinstance(shape, (list, tuple)):
            value, minv, maxv = random_variable(shape, dtype)
            return Argument(value, ArgType.VARIABLE, minv, maxv, shape, dtype)
        else:
            raise Exception('Unknown signature {0}'.format(signature))
    if label == "nparray":
        shape = signature['shape']
        dtype = signature['dtype']
        value, minv, maxv = random_tensor(shape, dtype)
        return Argument(value, ArgType.TENSOR, minv, maxv, shape, dtype)

    return Argument(None, ArgType.NULL)


class TFAPI:
    def __init__(self, filename: str=None, record=None, api_name=None) -> None:
        self.record = get_rand_single_signature(filename) if record is None else record
        self.api_name = TFAPI.file_to_apistr(filename) if api_name is None else api_name
        self.args = TFAPI.generate_args_from_record(self.record, self.api_name)
        self.filename = filename

    def get_number_of_arguments(self):
        arg_names = self.args.keys()
        return len(arg_names)

    def random_arg_names(self, mutate_arg_cnt):
        arg_names = list(self.args.keys())
        mutate_arg_names = choice(arg_names, mutate_arg_cnt)
        return mutate_arg_names

    def call(self) -> bool:
        args = []
        kwargs = {}
        inputs = None
        for key in self.args.keys():
            if "parameter:" in key:
                if self.args[key].type == ArgType.NULL: continue
                if self.args[key].type == ArgType.TF_OBJECT: continue
                args.append(self.args[key].get_value())
            elif key == "input_signature":
                inputs = self.args[key].get_value()
            elif key != "output_signature":
                if self.args[key].type == ArgType.NULL: continue
                if self.args[key].type == ArgType.TF_OBJECT: continue
                kwargs[key] = self.args[key].get_value()

        try:
            func = eval(self.api_name)
            if inspect.isclass(func):
                obj = func(*args, **kwargs)
                if inputs:
                    obj(*inputs)
            else:
                result = func(*args, **kwargs)

        except Exception as e:
            print(
                colored("[failed] api = %s" % (self.api_name),
                        "red"))
            print(colored(e, "red"))
            code = self.to_code()
            return False
        else:
            print(
                colored("[passed] api = %s" % (self.api_name),
                        "green"))

            code = self.to_code()
            return True

    def mutate_type(self, mutate_arg_names):
        def if_mutate_tensor():
            return random.random() < 0.9
        def if_mutate_null():
            return random.random() < 0.01
        def if_mutate_other():
            return random.random() < 0.1
        for arg_name in mutate_arg_names:
            self.args[arg_name].mutate_type()
            if self.args[arg_name].type == ArgType.TENSOR and if_mutate_tensor():
                self.args[arg_name].mutate_type()
            elif self.args[arg_name].type == ArgType.NULL and if_mutate_null():
                self.args[arg_name].mutate_type()
            elif if_mutate_other():
                self.args[arg_name].mutate_type()

    def mutate_value_random(self, mutate_arg_names):
        for arg_name in mutate_arg_names:
            self.args[arg_name].mutate_value_random()

    def mutate_value_db(self, mutate_arg_names):
        # Returns the list of mutated argument names
        def if_mutate_value():
            return random.random() < 0.7
        mutated_args = []
        for arg_name in mutate_arg_names:
            if if_mutate_value():
                if self.args[arg_name].mutate_value_db(self.api_name, arg_name):
                    mutated_args.append(arg_name)
        return mutated_args

    @staticmethod
    def file_to_apistr(filepath) -> str:
        """
        given a filepath (may include path and txt), convert it into an API name
        """
        filepath = filepath.strip()
        _, filename = os.path.split(filepath)
        filename = "tf." + filename.replace(".txt", "")
        return filename

    @staticmethod
    def generate_args_from_record(record: dict, api_name: str = ""): 

        def generate_args_from_signatures(signatures):
            if isinstance(signatures, dict):
                if signatures['Label'] == 'list':
                    s = signatures['value']
                    if isinstance(s, list):
                        signatures = s
            args = []
            for signature in signatures:
                x = generate_arg_from_signature(signature)
                args.append(x)
            return args

        args = {}
        for key in record.keys():
            if key == "input_signature":
                value = generate_args_from_signatures(record[key])
                args[key] = Argument(value, ArgType.LIST)
            elif key != "output_signature":
                args[key] = generate_arg_from_signature(record[key])
        return args

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
        func = eval(self.api_name)
        for arg in args:
            code += arg.to_code("arg_%d" % (index))
            arg_str += "arg_%d," % (index)
            index += 1
        for key, arg in kwargs.items():
            if arg.type == ArgType.NULL: continue
            if arg.type == ArgType.TF_OBJECT: continue
            code += arg.to_code(key)
            arg_str += "%s=%s," % (key, key)

        if inspect.isclass(func):
            code += "cls = %s(%s)\n" % (self.api_name, arg_str)
            if inputs:
                code += inputs.to_code("ins")
                code += "res = cls(*ins)\n"

        else:
            code += "res = %s(%s)\n" % (self.api_name, arg_str)
        return code