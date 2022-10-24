import gpt_2_simple as gpt2
import tensorflow as tf
import tweepy as tweepy
import os


try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# Authenticate to Twitter
consumer_key = os.getenv('CONSUMER_KEY')
consumer_secret = os.getenv('CONSUMER_SECRET')
access_token = os.getenv('ACCESS_TOKEN')
access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')
bearer_token = os.getenv('BEARER_TOKEN')


client = tweepy.Client(bearer_token,consumer_key,consumer_secret,access_token,access_token_secret)

gpt2.download_gpt2(model_name="124M")

file_name = "train_data.csv"

sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=file_name,
              model_name='124M',
              steps=50,
              restore_from='fresh',
              run_name='run1',
              print_every=10,
              sample_every=500,
              save_every=500,
	          only_train_transformer_layers = True,
	          accumulate_gradients = 1
              )




gen_file = 'gentext.txt'

gpt2.generate_to_file(sess,
                      destination_path=gen_file,
                      length=180,
                      temperature=1.0,
                      top_p=0.9,
                      prefix='<|startoftext|>',
                      truncate='<|endoftext|>',
                      include_prefix=False,
                      nsamples=1,
                      batch_size=1
                      )


f = open("gentext.txt", "r")

response = client.create_tweet(text=f.read())
print(str(response))

