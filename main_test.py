trained_model = BaseEcogFingerflexModel.load_from_checkpoint(
    checkpoint_path=model_to_test,
    model=AutoEncoder1D(**hp_autoencoder))
test_callback = TestCallback(ecog_data_val, fingerflex_data_val, finger_num)
test_callback.test(trained_model) # Testing