def build_conveyor(
    config_path: str,
    config_handle_function,
    methods_mapping,
    build_pipeline,
    run_function
):
    config = config_handle_function(config_path)
    kwargs = methods_mapping(config)
    pipeline = build_pipeline(**kwargs)
    run_function(pipeline)
