from fractions import Fraction
from typing import Dict, Union
import z3

ModelDict = Dict[str, Union[Fraction, bool]]


def model_to_dict(model: z3.ModelRef) -> ModelDict:
    ''' Utility function that takes a z3 model and extracts its variables to a
    dict'''
    decls = model.decls()
    res: Dict[str, Union[float, bool]] = {}
    for d in decls:
        val = model[d]
        if type(val) == z3.BoolRef:
            res[d.name()] = bool(val)
        elif type(val) == z3.IntNumRef:
            res[d.name()] = val.as_long()
        else:
            # Assume it is numeric
            decimal = val.as_decimal(100)
            if decimal[-1] == '?':
                decimal = decimal[:-1]
            res[d.name()] = float(decimal)
    return res
