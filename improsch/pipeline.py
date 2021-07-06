from typing import Callable, Optional
from functools import partial
from .processors import Deduplicator, resize_batch
from .filters import get_filter_by_min_size
from .readers import get_image_reader


def chain_functions(*functions) -> Callable:
    def func(*init_args, **init_kwargs):
        intermidiate_result = functions[0](*init_args, **init_kwargs)

        for next_function in functions[1:]:
            next_function(intermidiate_result)

        return intermidiate_result

    return func


def build_pipeline(
    filter_by_size: bool,
    need_resize: bool,
    deduplication: bool,
    filter_args: Optional[dict] = None,
    resize_args: Optional[dict] = None,
    deduplication_args: Optional[dict] = None
):
    if filter_by_size:
        if not filter_args:
            raise AttributeError("If set flag filter_by_size you have to provide filter_args!")

        filter_func = get_filter_by_min_size(**filter_args)
    else:
        filter_func = None

    reader = get_image_reader(filter_func)

    processing_functions = [reader]

    if need_resize:
        if not resize_args:
            raise AttributeError("If set flag need_resize you have to provide resize_args!")

        resize_partial = partial(resize_batch, **resize_args)
        processing_functions.append(resize_partial)

    if deduplication:
        if not deduplication_args:
            raise AttributeError("If set flag deduplication you have to provide deduplication_args!")

        deduplicator = Deduplicator(8)  # TODO read pool_size from config
        deduplicate_partial = partial(deduplicator, **deduplication_args)
        processing_functions.append(deduplicate_partial)

    return chain_functions(*processing_functions)
