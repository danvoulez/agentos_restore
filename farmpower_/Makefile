init_institution:
	logline run spans/setup_model.logline --simulate
	logline collapse spans/simulado_setup.logline
	logline run spans/setup_model.logline

train_epoch:
	for epoch in {1..10}; do \
		logline simulate spans/epoch_$${epoch}.logline ; \
		logline collapse spans/simulado_epoch_$${epoch}.logline ; \
		logline run spans/epoch_$${epoch}.logline --executor=tensor_engine ; \
		logline commit spans/checkpoint_$${epoch}.logline ; \
	done

emergency_rollback:
	logline compensate spans/checkpoint_$${epoch}.logline