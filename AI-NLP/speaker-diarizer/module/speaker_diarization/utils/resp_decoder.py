import pickle


def response_decoder(resp):

    csv_path = "../results/test.csv"

    # response decoder, stores the dataframe and the audio chunks to the system.
    # Extract the audio files corresponding to users.
    # Decode and save csv.
    # NOTE : The csv_path and the chunk_storage paths can be configured by the api team, depending upon its usage.
    df = pickle.loads(resp["Speaker_Diarizer_Result"]["Details"]["dataframe"])
    df.to_csv(csv_path)

    segment_dict = resp["Speaker_Diarizer_Result"]["Details"]["audio_chunks"]

    for spk in segment_dict.keys():
        for chunk_name in segment_dict[spk].keys():
            audio_file = pickle.loads(segment_dict[spk][chunk_name])
            audio_file.export(f"./{chunk_name}", format="wav")
