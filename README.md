# Question Answering on Multiparty Dialogue

Question Answering challenges the machine's ability of answering to queries in different forms.
This is a part of the [Character Mining](../../../character-mining) project led by the [Emory NLP](http://nlp.mathcs.emory.edu) research group.
The following shows a multiparty dialogue between Joey and Chandler with 6 questions regarding the contents of the dialogue. Questions could be in 6 forms (_what, who, when, where, why, how_)

| Speaker | Utterance |
|:-------:|-----------|
| `U01` | [Scene: Central Perk, Joey is getting a phone number from a woman (Casey) as Chandler watches from the doorway.] |
| `U02` | Casey: Here you go. |
| `U03` | Joey: Great! All right, so I’ll call you later.|
| `U04` | Casey: Great!|
| `U05` | Chandler: Hey-Hey-Hey! Who was that? |
| `U06` | Joey: That would be Casey. We’re going out tonight. |
| `U07` | Chandler: Goin’ out, huh? Wow! Wow! So things didn’t work out with Kathy, huh? Bummer.|
| `U08` | Joey: No, things are fine with Kathy. I’m having a late dinner with her tonight, right after my early dinner with Casey. |
| `U09` | Chandler: What? |
| `U10` | Joey: Yeah-yeah. And the craziest thing is that I just ate a whole pizza by myself! |
| `U11` | Chandler: Wait! You’re going out with Kathy! |
| `U12` | Joey: Yeah. Why are you getting so upset? |
| `U13` | Chandler: Well, I’m upset for you. I mean, dating an endless line of beautiful women must be very unfulfilling for you. |

<p align="left">
Q1: <code>What</code> is Joey going to do with Casey tonight?
</p>
<p align="left">
Q2: <code>Who</code> is Joey getting a phone number from?
</p>
<p align="left">
Q3: <code>When</code> will Joey have dinner with Kathy?
<p align="left">
Q4: <code>Where</code> are Joey and Chandler?
</p>
<p align="left">
Q5: <code>Why</code> is Chandler upset?
</p>
<p align="left">
Q6: <code>How</code> are things between Joey and Kathy?
</p>


Your task is to answering these open-domain questions using contiguous spans from the dialogues. 
This task is challenging because questions could be in any form and might not contain the exact words from the document. 

## Dataset

For the generation of the FriendsQA dataset, 1,222 scenes from the first four seasons of the Character Mining dataset are selected. Scenes with fewer than five utterances are discarded (83 of them), and each scene is considered an independent dialogue. FriendQA can be viewed as answer span selection, where questions are asked for some contexts in a dialogue and the model is expected to find certain spans in the dialogue containing answer contents. The dialogue aspects of this dataset, however, make it more challenging than other datasets comprising passages in formal languages. Details could be found in the paper. 

* Latest release: [v1.0](https://github.com/emorynlp/reading-comprehension/archive/reading-comprehension-1.0.tar.gz).
* [Release notes](https://github.com/emorynlp/reading-comprehension/releases).

## Statistics

Queries are randomly distributed to the Training (TRN), Development (DEV) and Test (TST) sets.

<!-- * U / Q: the average number of utterances per query.
* {E} / Q: the average number of entity types per query.
* [E] / Q: the average number of entities per query.
* {E} / U: the average number of entity types per utterance.
* [E] / U: the average number of entities per utterance. -->

| Dataset | Dialogues | Questions | Answers |
|:-------:|----------:|----------:|--------:|
| TRN     | 977       | 8,535     | 17,074  |
| DEV     | 122       | 1,010     | 2,057   |
| TST     | 123       | 1,065     | 2,131   |
| Total   | 1,222     | 10,610    | 21,262  |


## Annotation

Every scene has utterances concatenated together with 8-12 questions. 

```json
{
      "title": "s04_e19_c04",
      "paragraphs": [
        {
          "context": "u001 [ Scene : Central Perk , Joey is whining to Chandler about the tickets . ] u002 Joey_Tribbiani Come on ! u003 Chandler_Bing Yes , Gunther , can I get two cups of chino , please ? u004 Gunther Good one . u005 Joey_Tribbiani Come on , season tickets ! Season tickets , do you know what that means ? u006 Chandler_Bing Forget it ! Okay , I 'm not giving up the apartment . u007 Joey_Tribbiani Oh come - look , when I was a kid my dad 's company gave season tickets to the number one salesman every year , all right ? My dad never won ! Of course , he was n't in the sales division , but still , I never ever , ever forgot that ! u008 Ross_Geller Hey , guys ! u009 Joey_Tribbiani Hey ! u010 Chandler_Bing Oh my God ! u011 Joey_Tribbiani We do n't make enough fun of you already ? u012 Ross_Geller Oh yeah , Emily convinced me to do it . u013 Chandler_Bing You do know that Wham broke up ? u014 Ross_Geller I like it , and Emily likes it , and that 's what counts . So uh , how are you guys doing ? u015 Joey_Tribbiani Oh - no , do n't try and talk all normal with that thing in your ear . u016 Chandler_Bing Where is Emily ? u017 Ross_Geller Ugh , she 's saying good - bye to her uncle . u018 Chandler_Bing Man , did n't she like just get here ? u019 Ross_Geller Yeah !! Yeah ! u020 Chandler_Bing Easy tiger . u021 Ross_Geller I just , I hate this so much ! I mean , every time I go pick her up at the airport , it 's - it 's so great . But at the same time I 'm thinking , \" Well , I 'm gon na be right back there in a couple of days , dropping her off . \" u022 Chandler_Bing So what are you going to do ? u023 Ross_Geller Nothing ! There 's nothing to do ! I mean , she lives there , I live here . I mean , she - she 'd have to uh , move here . She should move here ! u024 Joey_Tribbiani What ? u025 Ross_Geller I could ask her to live with me ! u026 Chandler_Bing Are you serious ? u027 Ross_Geller I mean , why not ! I mean , I mean why not ?! u028 Chandler_Bing Because you 've only known her for six weeks ! Okay , I 've got a carton of milk in my fridge I 've had a longer relationship with ! u029 Ross_Geller Look guys , when I 'm with her it 's - it 's - it 's like she brings this - this - this great side out of me . I mean I-I-I love her , y'know ? u030 Chandler_Bing And I love the milk ! But , I 'm not gon na some British girl to move in with me ! Joey , you say things now . u031 Joey_Tribbiani All right look , Ross , he 's right . Emily 's great , she 's great ! But this way too soon , you 're only gon na scare her ! u032 Ross_Geller I do n't want to do that . u033 Joey_Tribbiani No ! You do n't want to wreck it , you do n't want to go to fast ! u034 Ross_Geller Yeah , no , you 're right , I know , you 're right , I 'm not , I 'm not gon na do it . All right , thanks guys . u035 Chandler_Bing Okay , no problem , just remember to wake us up before you go - go .",
          "qas": [
            {
              "answers": [
                {
                  "answer_start": 30,
                  "answer_end": 45,
                  "text": "Joey is whining",
                  "utterance_id": 0,
                  "utterance_start": "u001"
                },
                {
                  "answer_start": 38,
                  "answer_end": 57,
                  "text": "whining to Chandler",
                  "utterance_id": 0,
                  "utterance_start": "u001"
                }
              ],
              "id": "s04_e19_c04_What",
              "question": "What action is Joey doing ?"
            },
            {
              "answers": [
                {
                  "answer_start": 115,
                  "answer_end": 128,
                  "text": "Chandler_Bing",
                  "utterance_id": 110,
                  "utterance_start": "u003"
                },
                {
                  "answer_start": 115,
                  "answer_end": 128,
                  "text": "Chandler_Bing",
                  "utterance_id": 110,
                  "utterance_start": "u003"
                }
              ],
              "id": "s04_e19_c04_Who",
              "question": "Who asks for two cups of chino ?"
            },
            {
              "answers": [
                {
                  "answer_start": 15,
                  "answer_end": 27,
                  "text": "Central Perk",
                  "utterance_id": 0,
                  "utterance_start": "u001"
                },
                {
                  "answer_start": 15,
                  "answer_end": 27,
                  "text": "Central Perk",
                  "utterance_id": 0,
                  "utterance_start": "u001"
                }
              ],
              "id": "s04_e19_c04_Where",
              "question": "Where are they ?"
            },
            {
              "answers": [
                {
                  "answer_start": 64,
                  "answer_end": 77,
                  "text": "the tickets .",
                  "utterance_id": 0,
                  "utterance_start": "u001"
                },
                {
                  "answer_start": 58,
                  "answer_end": 75,
                  "text": "about the tickets",
                  "utterance_id": 0,
                  "utterance_start": "u001"
                }
              ],
              "id": "s04_e19_c04_Why",
              "question": "Why is Joey whining ?"
            },
            {
              "answers": [
                {
                  "answer_start": 808,
                  "answer_end": 837,
                  "text": "Emily convinced me to do it .",
                  "utterance_id": 781,
                  "utterance_start": "u012"
                },
                {
                  "answer_start": 798,
                  "answer_end": 837,
                  "text": "Oh yeah , Emily convinced me to do it .",
                  "utterance_id": 781,
                  "utterance_start": "u012"
                }
              ],
              "id": "s04_e19_c04_How",
              "question": "How was Ross convinced ?"
            },
            {
              "answers": [
                {
                  "answer_start": 30,
                  "answer_end": 45,
                  "text": "Joey is whining",
                  "utterance_id": 0,
                  "utterance_start": "u001"
                },
                {
                  "answer_start": 38,
                  "answer_end": 57,
                  "text": "whining to Chandler",
                  "utterance_id": 0,
                  "utterance_start": "u001"
                }
              ],
              "id": "s04_e19_c04_What_Paraphrased",
              "question": "What is Joey doing about the tickets ?"
            },
            {
              "answers": [
                {
                  "answer_start": 115,
                  "answer_end": 128,
                  "text": "Chandler_Bing",
                  "utterance_id": 110,
                  "utterance_start": "u003"
                },
                {
                  "answer_start": 115,
                  "answer_end": 128,
                  "text": "Chandler_Bing",
                  "utterance_id": 110,
                  "utterance_start": "u003"
                }
              ],
              "id": "s04_e19_c04_Who_Paraphrased",
              "question": "Who order two cups of Chino from Gunther ?"
            },
            {
              "answers": [
                {
                  "answer_start": 15,
                  "answer_end": 27,
                  "text": "Central Perk",
                  "utterance_id": 0,
                  "utterance_start": "u001"
                },
                {
                  "answer_start": 15,
                  "answer_end": 27,
                  "text": "Central Perk",
                  "utterance_id": 0,
                  "utterance_start": "u001"
                }
              ],
              "id": "s04_e19_c04_Where_Paraphrased",
              "question": "Where are Chandler and Joey ?"
            },
            {
              "answers": [
                {
                  "answer_start": 64,
                  "answer_end": 77,
                  "text": "the tickets .",
                  "utterance_id": 0,
                  "utterance_start": "u001"
                },
                {
                  "answer_start": 58,
                  "answer_end": 75,
                  "text": "about the tickets",
                  "utterance_id": 0,
                  "utterance_start": "u001"
                }
              ],
              "id": "s04_e19_c04_Why_Paraphrased",
              "question": "Why is Joey whining to Chandler ?"
            },
            {
              "answers": [
                {
                  "answer_start": 808,
                  "answer_end": 837,
                  "text": "Emily convinced me to do it .",
                  "utterance_id": 781,
                  "utterance_start": "u012"
                },
                {
                  "answer_start": 798,
                  "answer_end": 837,
                  "text": "Oh yeah , Emily convinced me to do it .",
                  "utterance_id": 781,
                  "utterance_start": "u012"
                }
              ],
              "id": "s04_e19_c04_How_Paraphrased",
              "question": "How did Ross say he was convinced ?"
            }
          ]
        }
      ]
    }
```

## Citation

* [FriendsQA: Open-Domain Question Answering on TV Show Transcripts](https://www.aclweb.org/anthology/W19-5923). Zhengzhe Yang and Jinho D. Choi. In Proceedings of the Annual Conference of the ACL Special Interest Group on Discourse and Dialogue, SIGDIAL'19, 2019 ([slides](https://www.slideshare.net/jchoi7s/friendsqa-opendomain-question-answering-on-tv-show-transcripts-154329602)).

## Contact

* [Jinho D. Choi](http://www.mathcs.emory.edu/~choi).
