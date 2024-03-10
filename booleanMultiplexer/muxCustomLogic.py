def generate_all_input_combinations_for_model(process_id, process_address_map):
    all_combinations = []
    for x in range(256):
        for address_to_approximate in process_address_map[process_id]:
            process_id_bin_string = '{0:03b}'.format(address_to_approximate)
            all_combinations.append(process_id_bin_string + '{0:08b}'.format(
                x))  # https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
    return all_combinations
