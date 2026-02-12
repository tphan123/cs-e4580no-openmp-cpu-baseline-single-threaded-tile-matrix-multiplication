import os
from typing import Optional
from ppcgrader.compiler import Compiler
import ppcgrader.config


class Config(ppcgrader.config.Config):
    def __init__(self,
                 code: str,
                 openmp: bool = False,
                 gpu: bool = False,
                 vnni: bool = False,
                 turing: bool = False):
        from . import info
        super().__init__(binary='i8mm',
                         cfg_file=__file__,
                         info=info,
                         gpu=gpu,
                         openmp=openmp,
                         code=code)
        self.vnni = vnni
        self.turing = turing

    def common_flags(self, compiler: Compiler) -> Compiler:
        compiler = super().common_flags(compiler)
        if self.gpu:
            compiler = compiler.add_flag("-lineinfo")
            if self.turing:
                compiler = compiler.add_flag("-arch=native")
        if not self.vnni and not self.gpu:
            compiler = compiler.add_flag("-mno-avx512vnni")
        return compiler

    def parse_output(self, output):
        input_data = {
            "n": None,
            "m": None,
            "k": None,
            "input_a": None,
            "input_b": None,
            "tile_size": None,
        }
        output_data = {
            "result": None,
        }
        output_errors = {
            "locations": None,
        }
        statistics = {}

        def parse_matrix(string):
            M = string.strip('[]').split(';')
            M = [row.strip() for row in M]
            M = [row.split(" ") for row in M]
            M = [[int(e) for e in row] for row in M]
            return M

        for line in output.splitlines():
            splitted = line.split('\t')
            if splitted[0] == 'result':
                errors = {
                    'fail': True,
                    'pass': False,
                    'done': False
                }[splitted[1]]
            elif splitted[0] == 'time':
                time = float(splitted[1])
            elif splitted[0] == 'perf_wall_clock_ns':
                time = int(splitted[1]) / 1e9
                statistics[splitted[0]] = int(splitted[1])
            elif splitted[0].startswith('perf_'):
                statistics[splitted[0]] = int(splitted[1])
            elif splitted[0] in ['n', 'm', 'k', 'tile_size']:
                input_data[splitted[0]] = int(splitted[1])
            elif splitted[0] in ['input_a', 'input_b']:
                input_data[splitted[0]] = parse_matrix(splitted[1])
            elif splitted[0] == 'output':
                output_data["result"] = parse_matrix(splitted[1])
            elif splitted[0] == 'locations':
                output_errors["locations"] = parse_matrix(splitted[1])

        m = input_data.get('m', None)
        n = input_data.get('n', None)
        k = input_data.get('k', None)
        if m and n and k:
            flops = 2 * m * n * k
            statistics['operations'] = flops
            statistics['operations_name'] = "useful arithmetic operation"

        return time, errors, input_data, output_data, output_errors, statistics
