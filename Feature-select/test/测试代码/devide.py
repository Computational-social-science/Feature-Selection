import numpy as np
import  pandas as pd
txt = "𝑈 𝑓1 𝑓2 𝑓3 𝑓4 𝑓5 𝑓6 𝑓7 𝑓8 𝐶 𝑥1 pretentious proper completed 2 convenient convenient non-prob not_recom not_recom 𝑥2 usual improper completed 2 convenient convenient slightly_prob not_recom not_recom 𝑥3 usual improper incomplete 3 convenient inconv non-prob not_recom not_recom 𝑥4 pretentious improper completed 3 convenient convenient non-prob recommended priority 𝑥5 usual proper completed 2 convenient convenient problematic recommended priority 𝑥6 usual improper completed 2 critical inconv non-prob recommended priority 𝑥7 pretentious proper complete 2 convenient convenient non-prob recommended very_recom 𝑥8 usual improper completed 2 convenient inconv non-prob recommended very_recom 𝑥9 usual improper complete 1 less_conv inconv non-prob recommended very_recom 𝑥10 usual improper complete 2 convenient convenient problematic recommended very_recom"

divide = txt.split( )
divide = np.array(divide)
# print(divide)
divide.reshape(11,10)
divide = pd.DataFrame(data=divide)
divide.to_csv('aaaaa.csv',encoding='utf_8_sig')
print(divide)
# np.savetxt(, divide,)