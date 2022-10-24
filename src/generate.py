import gpt_2_simple as gpt2

gpt2.download_gpt2(model_name="355M")

file_name = "train_data.csv"

sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=file_name,
              model_name='355M',
              steps=2000,
              restore_from='fresh',
              run_name='run1',
              print_every=10,
              sample_every=500,
              save_every=500
              )

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, run_name='run1')


gen_file = 'gentext.txt'

gpt2.generate_to_file(sess,
                      destination_path=gen_file,
                      length=200,
                      temperature=1.0,
                      top_p=0.9,
                      prefix='<|startoftext|>',
                      truncate='<|endoftext|>',
                      include_prefix=False,
                      nsamples=1,
                      batch_size=20
                      )
