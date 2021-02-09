import argparse
import sys
import inspect
import itertools
import fnmatch


def _add_arguments_for_module(parser,
                              module,
                              name,
                              default_class,
                              add_class_argument=True,   # whether to add class choice as argument
                              include_classes="*",
                              exclude_classes=[],
                              exclude_params=["self","args"],
                              param_defaults={},          # allows to overwrite any default param
                              forced_default_types={},    # allows to set types for known arguments
                              unknown_default_types={}):  # allows to set types for unknown arguments

    # -------------------------------------------------------------------------
    # Determine possible choices from class names in module, possibly apply include/exclude filters
    # -------------------------------------------------------------------------
    module_dict = module_classes_to_dict(
        module, include_classes=include_classes, exclude_classes=exclude_classes)

    # -------------------------------------------------------------------------
    # Parse known arguments to determine choice for argument name
    # -------------------------------------------------------------------------
    if add_class_argument:
        parser.add_argument(
            "--%s" % name, type=str, default=default_class, choices=module_dict.keys())
        known_args = parser.parse_known_args(sys.argv[1:])[0]
    else:
        # build a temporary parser, and do not add the class as argument
        tmp_parser = argparse.ArgumentParser()
        tmp_parser.add_argument(
            "--%s" % name, type=str, default=default_class, choices=module_dict.keys())
        known_args = tmp_parser.parse_known_args(sys.argv[1:])[0]

    class_name = vars(known_args)[name]

    # -------------------------------------------------------------------------
    # If class is None, there is no point in trying to parse further arguments
    # -------------------------------------------------------------------------
    if class_name is None:
        return

    # -------------------------------------------------------------------------
    # Get constructor of that argument choice
    # -------------------------------------------------------------------------
    class_constructor = module_dict[class_name]

    # -------------------------------------------------------------------------
    # Determine constructor argument names and defaults
    # -------------------------------------------------------------------------
    try:
        argspec = inspect.getargspec(class_constructor.__init__)
        argspec_defaults = argspec.defaults if argspec.defaults is not None else []
        full_args = argspec.args
        default_args_dict = dict(zip(argspec.args[-len(argspec_defaults):], argspec_defaults))
    except TypeError:
        print(argspec)
        print(argspec.defaults)
        raise ValueError("unknown_default_types should be adjusted for module: '%s.py'" % name)

    # -------------------------------------------------------------------------
    # Add sub_arguments
    # -------------------------------------------------------------------------
    for argname in full_args:

        # ---------------------------------------------------------------------
        # Skip
        # ---------------------------------------------------------------------
        if argname in exclude_params:
            continue

        # ---------------------------------------------------------------------
        # Sub argument name
        # ---------------------------------------------------------------------
        sub_arg_name = "%s_%s" % (name, argname)

        # ---------------------------------------------------------------------
        # If a default argument is given, take that one
        # ---------------------------------------------------------------------
        if argname in param_defaults.keys():
            parser.add_argument(
                "--%s" % sub_arg_name,
                type=_get_type_from_arg(param_defaults[argname]),
                default=param_defaults[argname])

        # ---------------------------------------------------------------------
        # If a default parameter can be inferred from the module, pick that one
        # ---------------------------------------------------------------------
        elif argname in default_args_dict.keys():

            # -----------------------------------------------------------------
            # Check for forced default types
            # -----------------------------------------------------------------
            if argname in forced_default_types.keys():
                argtype = forced_default_types[argname]
            else:
                argtype = _get_type_from_arg(default_args_dict[argname])
            parser.add_argument(
                "--%s" % sub_arg_name, type=argtype, default=default_args_dict[argname])

        # ---------------------------------------------------------------------
        # Take from the unkowns list
        # ---------------------------------------------------------------------
        elif argname in unknown_default_types.keys():
            parser.add_argument("--%s" % sub_arg_name, type=unknown_default_types[argname])

        else:
            raise ValueError(
                "Do not know how to handle argument '%s' for class '%s'" % (argname, name))


def module_classes_to_dict(module, include_classes="*", exclude_classes=()):
    # -------------------------------------------------------------------------
    # If arguments are strings, convert them to a list
    # -------------------------------------------------------------------------
    if include_classes is not None:
        if isinstance(include_classes, str):
            include_classes = [include_classes]

    if exclude_classes is not None:
        if isinstance(exclude_classes, str):
            exclude_classes = [exclude_classes]

    # -------------------------------------------------------------------------
    # Obtain dictionary from given module
    # -------------------------------------------------------------------------
    item_dict = dict([(name, getattr(module, name)) for name in dir(module)])

    # -------------------------------------------------------------------------
    # Filter classes
    # -------------------------------------------------------------------------
    item_dict = dict([
        (name,value) for name, value in item_dict.items() if inspect.isclass(getattr(module, name))
    ])

    filtered_keys = filter_list_of_strings(
        item_dict.keys(), include=include_classes, exclude=exclude_classes)

    # -------------------------------------------------------------------------
    # Construct dictionary from matched results
    # -------------------------------------------------------------------------
    result_dict = dict([(name, value) for name, value in item_dict.items() if name in filtered_keys])

    return result_dict


def filter_list_of_strings(lst, include="*", exclude=()):
    filtered_matches = list(itertools.chain.from_iterable([fnmatch.filter(lst, x) for x in include]))
    filtered_nomatch = list(itertools.chain.from_iterable([fnmatch.filter(lst, x) for x in exclude]))
    matched = list(set(filtered_matches) - set(filtered_nomatch))
    return matched


def _get_type_from_arg(arg):
    if isinstance(arg, bool):
        return str2bool
    else:
        return type(arg)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2intlist(v):
    return [int(x.strip()) for x in v.strip()[1:-1].split(',')]
