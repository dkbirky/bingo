"""
This module represents the python backend associated with the Agraph equation
evaluation.  This backend is used to perform the the evaluation of the equation
represented by an `AGraph`.  It can also perform derivatives.
"""
import numpy as np

import torch
from torch.func import hessian as get_hessian
from torch_eval import evaluate as cpp_evaluate
from torch_eval import evaluate_with_deriv as cpp_evaluate_with_deriv
from .operator_eval import forward_eval_function
from torch.autograd import grad
ENGINE = "pytorch_cpp"


def _get_torch_const(constants, data_len):
    with torch.no_grad():
        constants = torch.from_numpy(np.array(constants)).double()
        if len(constants) > 0:
            if constants.ndim == 1:
                constants = constants.unsqueeze(1)
            constants = constants.unsqueeze(2).expand(-1, -1, data_len).mT
        return constants


def _reshape_output(output, constants, x):
    x_dim = x.size(1)
    c_dim = 1
    if len(constants) > 0:
        if isinstance(constants[0], np.ndarray):
            c_dim = len(constants[0])
    if isinstance(output, np.ndarray):
        if output.shape == (x_dim, c_dim):
            return output
        elif output.shape == (x_dim,):
            return output.reshape((x_dim, 1))
    return np.ones((x_dim, c_dim)) * output


def get_pytorch_repr(command_array):
    # TODO see if we can do this more efficiently
    # TODO this reruns on every eval, how to return just expression?

    def get_expr(X, constants):  # assumes X is column-order
        expr = []

        for (node, param1, param2) in command_array:
            expr.append(forward_eval_function(node, param1, param2, X, constants,
                                              expr))

        return expr[-1]

    return get_expr


def evaluate(pytorch_repr, x, constants, final=True):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.T).double()
    if final:
        constants = _get_torch_const(constants, x.size(1))
    return_eval = get_pytorch_repr(pytorch_repr)(x, constants)
    if final:
        return _reshape_output(return_eval.detach().numpy(), constants, x)
    return return_eval


def evaluate_with_derivative(cmd_arr, x, constants, wrt_param_x_or_c):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.T).double()
    torch_constants = _get_torch_const(constants, x.size(1))
    eval, deriv = cpp_evaluate_with_deriv(cmd_arr, x, torch_constants, wrt_param_x_or_c)
    if deriv.ndim == 3:
        deriv = deriv[:, :, 0]
    return _reshape_output(eval.detach().numpy(), constants, x), deriv.T.detach().numpy()


def evaluate_with_hessian(cmd_arr, x, constants, wrt_param_x_or_c):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.T).double()
    evaluation = evaluate(cmd_arr, x, constants)

    torch_constants = torch.from_numpy(np.array(constants)).double()

    deriv_argnum = 2  # wrt c
    if wrt_param_x_or_c:  # wrt x
        deriv_argnum = 1
    hessian = get_hessian(cpp_evaluate, argnums=deriv_argnum)(cmd_arr, x,
                                                              torch_constants)

    return evaluation, hessian

def evaluate_with_partials(pytorch_repr, x, constants, partial_order):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.T).double()
    if not isinstance(constants, np.ndarray):
        constants = np.array(constants).T

    final_eval = evaluate(pytorch_repr, x, constants, final=True)

    x = x[:, :, None].expand(-1, -1, constants.shape[1])
    x.requires_grad = True
    constants = _get_torch_const(constants, x.size(1))
    eval = evaluate(pytorch_repr, x, constants, final=False)

    partial = eval
    partials = []
    for variable in partial_order:
        try:
            partial = grad(outputs=partial.sum(), inputs=x,
                         allow_unused=True,
                         create_graph=True)[0][variable]
            if partial is None:
                partial = torch.zeros_like(x[0])
        except (IndexError, RuntimeError):
            partial = torch.zeros_like(x[0])
        partials.append(partial.detach().numpy())

    return _reshape_output(final_eval, constants, x), partials
