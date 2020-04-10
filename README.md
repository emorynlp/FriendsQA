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

* Q1: <code>What</code> is Joey going to do with Casey tonight?
* Q2: <code>Who</code> is Joey getting a phone number from?
* Q3: <code>When</code> will Joey have dinner with Kathy?
* Q4: <code>Where</code> are Joey and Chandler?
* Q5: <code>Why</code> is Chandler upset?
* Q6: <code>How</code> are things between Joey and Kathy?

Your task is to answering these open-domain questions using contiguous spans from the dialogues. 
This task is challenging because questions could be in any form and might not contain the exact words from the document. 

## Dataset

For the generation of the FriendsQA dataset, 1,222 scenes from the first four seasons of the Character Mining dataset are selected. Scenes with fewer than five utterances are discarded (83 of them), and each scene is considered an independent dialogue. FriendQA can be viewed as answer span selection, where questions are asked for some contexts in a dialogue and the model is expected to find certain spans in the dialogue containing answer contents. The dialogue aspects of this dataset, however, make it more challenging than other datasets comprising passages in formal languages. Details could be found in the paper. 

* Latest release: [v2.0](https://github.com/emorynlp/reading-comprehension/archive/reading-comprehension-2.0.tar.gz)

## Statistics

The data split is based on chronological order of the episodes that is consistent across other Character Mining projects:

| Dataset | Dialogues | Questions | Answers | Eposides |
| :-----: | --------: | --------: | ------: | -------: |
|   TRN   |       973 |     9,791 |  16,352 |   1 - 20 |
|   DEV   |       113 |     1,189 |   2,065 |  21 - 22 |
|   TST   |       136 |     1,172 |   1,920 |   23 - * |


## 

## Annotation

<!-- Below is the whole context data format: every scene has utterances concatenated together with 8-12 questions. 

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

Below is the split context data format: 

```json
"title": "s01_e23_c06",
            "paragraphs": [
                {
                    "utterances:": [
                        {
                            "uid": 0,
                            "speakers": [
                                "Ross Geller"
                            ],
                            "utterance": "Breathe ."
                        },
                        {
                            "uid": 1,
                            "speakers": [
                                "Susan Bunch"
                            ],
                            "utterance": "Breathe ."
                        },
                        {
                            "uid": 2,
                            "speakers": [
                                "Ross Geller"
                            ],
                            "utterance": "Breathe ."
                        },
                        {
                            "uid": 3,
                            "speakers": [
                                "Susan Bunch"
                            ],
                            "utterance": "Breathe ."
                        },
                        {
                            "uid": 4,
                            "speakers": [
                                "Ross Geller"
                            ],
                            "utterance": "Breathe ."
                        },
                        {
                            "uid": 5,
                            "speakers": [
                                "Susan Bunch"
                            ],
                            "utterance": "Breathe ."
                        },
                        {
                            "uid": 6,
                            "speakers": [
                                "Carol Willick"
                            ],
                            "utterance": "You 're gon na kill me !"
                        },
                        {
                            "uid": 7,
                            "speakers": [
                                "Ross Geller"
                            ],
                            "utterance": "15 more seconds , 14 , 13 , 12 ..."
                        },
                        {
                            "uid": 8,
                            "speakers": [
                                "Carol Willick"
                            ],
                            "utterance": "Count faster ."
                        },
                        {
                            "uid": 9,
                            "speakers": [
                                "Susan Bunch"
                            ],
                            "utterance": "It 's gon na be ok , just remember , we 're doing this for Jordie . Just keep focusing on Jordie ."
                        },
                        {
                            "uid": 10,
                            "speakers": [
                                "Ross Geller"
                            ],
                            "utterance": "Who the hell is Jordie ?"
                        },
                        {
                            "uid": 11,
                            "speakers": [
                                "Susan Bunch"
                            ],
                            "utterance": "Your son ."
                        },
                        {
                            "uid": 12,
                            "speakers": [
                                "Ross Geller"
                            ],
                            "utterance": "No - no - no . I do n't have a son named Jordie . We all agreed , my son 's name is Jamie ."
                        },
                        {
                            "uid": 13,
                            "speakers": [
                                "Carol Willick"
                            ],
                            "utterance": "Well , Jamie was the name of Susan 's first girlfriend , so we went back to Jordie ."
                        },
                        {
                            "uid": 14,
                            "speakers": [
                                "Ross Geller"
                            ],
                            "utterance": "What ? Whoa , whoa whoa whoa , what do you mean , back to Jordie ? We never landed on Jordie . We just passed by it during the whole Jessy , Cody , Dylan fiasco ."
                        },
                        {
                            "uid": 15,
                            "speakers": [
                                "Carol Willick"
                            ],
                            "utterance": "Ow , ow , ow , ow , leg cramp , leg cramp , leg cramp ."
                        },
                        {
                            "uid": 16,
                            "speakers": [
                                "Ross Geller"
                            ],
                            "utterance": "I got it ."
                        },
                        {
                            "uid": 17,
                            "speakers": [
                                "Susan Bunch"
                            ],
                            "utterance": "I got it ."
                        },
                        {
                            "uid": 18,
                            "speakers": [
                                "Ross Geller"
                            ],
                            "utterance": "I got it ! Hey , you get to sleep with her , I get the cramps ."
                        },
                        {
                            "uid": 19,
                            "speakers": [
                                "Susan Bunch"
                            ],
                            "utterance": "No , you do n't ."
                        },
                        {
                            "uid": 20,
                            "speakers": [
                                "Carol Willick"
                            ],
                            "utterance": "All right , that 's it . I want both of you out ."
                        },
                        {
                            "uid": 21,
                            "speakers": [
                                "Ross Geller"
                            ],
                            "utterance": "Why ?"
                        },
                        {
                            "uid": 22,
                            "speakers": [
                                "Susan Bunch"
                            ],
                            "utterance": "He started it !"
                        },
                        {
                            "uid": 23,
                            "speakers": [
                                "Ross Geller"
                            ],
                            "utterance": "No , you started it ."
                        },
                        {
                            "uid": 24,
                            "speakers": [
                                "Susan Bunch"
                            ],
                            "utterance": "You did !"
                        },
                        {
                            "uid": 25,
                            "speakers": [
                                "Carol Willick"
                            ],
                            "utterance": "I do n't care . I am trying to get a person out of my body here , and you 're not making it any easier ."
                        },
                        {
                            "uid": 26,
                            "speakers": [
                                "Ross Geller"
                            ],
                            "utterance": "But ..."
                        },
                        {
                            "uid": 27,
                            "speakers": [
                                "Carol Willick"
                            ],
                            "utterance": "Now go !"
                        },
                        {
                            "uid": 28,
                            "speakers": [
                                "Ross Geller"
                            ],
                            "utterance": "Thanks a lot ."
                        },
                        {
                            "uid": 29,
                            "speakers": [
                                "Susan Bunch"
                            ],
                            "utterance": "See what you did ."
                        },
                        {
                            "uid": 30,
                            "speakers": [
                                "Ross Geller"
                            ],
                            "utterance": "Yeah , listen ..."
                        },
                        {
                            "uid": 31,
                            "speakers": [
                                "Carol Willick"
                            ],
                            "utterance": "Out !"
                        },
                        {
                            "uid": 32,
                            "speakers": [
                                "#NOTE#"
                            ],
                            "utterance": "( Ross and Susan both angrily leave the hopsital room . )"
                        }
                    ],
                    "qas": [
                        {
                            "id": "s01_e23_c06_What",
                            "question": "What does Ross want to name his son ?",
                            "answers": [
                                {
                                    "answer_text": "Jamie",
                                    "utterance_id": 12,
                                    "inner_start": 24,
                                    "inner_end": 24,
                                    "is_speaker": false
                                },
                                {
                                    "answer_text": "Jordie .",
                                    "utterance_id": 9,
                                    "inner_start": 21,
                                    "inner_end": 22,
                                    "is_speaker": false
                                }
                            ]
                        },
                        {
                            "id": "s01_e23_c06_Who",
                            "question": "Who told Ross to count faster ?",
                            "answers": [
                                {
                                    "answer_text": "Carol Willick",
                                    "utterance_id": 6,
                                    "inner_start": -1,
                                    "inner_end": -1,
                                    "is_speaker": true
                                }
                            ]
                        },
                        {
                            "id": "s01_e23_c06_When",
                            "question": "When was the name Jordie passed by ?",
                            "answers": [
                                {
                                    "answer_text": "during the whole Jessy , Cody , Dylan fiasco",
                                    "utterance_id": 14,
                                    "inner_start": 28,
                                    "inner_end": 36,
                                    "is_speaker": false
                                },
                                {
                                    "answer_text": "No - no - no . I do n't have a son named Jordie",
                                    "utterance_id": 12,
                                    "inner_start": 0,
                                    "inner_end": 13,
                                    "is_speaker": false
                                }
                            ]
                        },
                        {
                            "id": "s01_e23_c06_Where",
                            "question": "Where are Susan Ross and Carol ?",
                            "answers": [
                                {
                                    "answer_text": "hopsital room",
                                    "utterance_id": 32,
                                    "inner_start": 8,
                                    "inner_end": 9,
                                    "is_speaker": false
                                },
                                {
                                    "answer_text": "the hopsital room",
                                    "utterance_id": 32,
                                    "inner_start": 7,
                                    "inner_end": 9,
                                    "is_speaker": false
                                }
                            ]
                        },
                        {
                            "id": "s01_e23_c06_Why",
                            "question": "Why did Carol not want to name the baby Jamie ?",
                            "answers": [
                                {
                                    "answer_text": "Jamie was the name of Susan 's first girlfriend",
                                    "utterance_id": 13,
                                    "inner_start": 2,
                                    "inner_end": 10,
                                    "is_speaker": false
                                }
                            ]
                        },
                        {
                            "id": "s01_e23_c06_How",
                            "question": "How did Ross and Susan leave the hospital room ?",
                            "answers": [
                                {
                                    "answer_text": "angrily",
                                    "utterance_id": 32,
                                    "inner_start": 5,
                                    "inner_end": 5,
                                    "is_speaker": false
                                }
                            ]
                        },
                        {
                            "id": "s01_e23_c06_What_Paraphrased",
                            "question": "What is the name Ross wishes for his son ?",
                            "answers": [
                                {
                                    "answer_text": "Jamie",
                                    "utterance_id": 12,
                                    "inner_start": 24,
                                    "inner_end": 24,
                                    "is_speaker": false
                                },
                                {
                                    "answer_text": "Jordie .",
                                    "utterance_id": 9,
                                    "inner_start": 21,
                                    "inner_end": 22,
                                    "is_speaker": false
                                }
                            ]
                        },
                        {
                            "id": "s01_e23_c06_Who_Paraphrased",
                            "question": "By whom was Ross told to count faster ?",
                            "answers": [
                                {
                                    "answer_text": "Carol Willick",
                                    "utterance_id": 6,
                                    "inner_start": -1,
                                    "inner_end": -1,
                                    "is_speaker": true
                                }
                            ]
                        },
                        {
                            "id": "s01_e23_c06_When_Paraphrased",
                            "question": "At what point was the name Jordie rejected ?",
                            "answers": [
                                {
                                    "answer_text": "during the whole Jessy , Cody , Dylan fiasco",
                                    "utterance_id": 14,
                                    "inner_start": 28,
                                    "inner_end": 36,
                                    "is_speaker": false
                                },
                                {
                                    "answer_text": "No - no - no . I do n't have a son named Jordie",
                                    "utterance_id": 12,
                                    "inner_start": 0,
                                    "inner_end": 13,
                                    "is_speaker": false
                                }
                            ]
                        },
                        {
                            "id": "s01_e23_c06_Where_Paraphrased",
                            "question": "What is Susan and Carols ' location ?",
                            "answers": [
                                {
                                    "answer_text": "hopsital room",
                                    "utterance_id": 32,
                                    "inner_start": 8,
                                    "inner_end": 9,
                                    "is_speaker": false
                                },
                                {
                                    "answer_text": "the hopsital room",
                                    "utterance_id": 32,
                                    "inner_start": 7,
                                    "inner_end": 9,
                                    "is_speaker": false
                                }
                            ]
                        },
                        {
                            "id": "s01_e23_c06_Why_Paraphrased",
                            "question": "Carol did not want to name the baby Jamie for what reason ?",
                            "answers": [
                                {
                                    "answer_text": "Jamie was the name of Susan 's first girlfriend",
                                    "utterance_id": 13,
                                    "inner_start": 2,
                                    "inner_end": 10,
                                    "is_speaker": false
                                }
                            ]
                        },
                        {
                            "id": "s01_e23_c06_How_Paraphrased",
                            "question": "In what fashion did Ross and Susan leave the hospital room ?",
                            "answers": [
                                {
                                    "answer_text": "angrily",
                                    "utterance_id": 32,
                                    "inner_start": 5,
                                    "inner_end": 5,
                                    "is_speaker": false
                                }
                            ]
                        }
                    ]
                }
            ]
        }
``` -->



## Citation

* [Transformers to Learn Hierarchical Contexts in Multiparty Dialogue for Span-based Question Answering](). Changmao Li and Jinho D. Choi. In Proceedings of the Conference of the Association for Computational Linguistics, ACL'20, 2020.
* [FriendsQA: Open-Domain Question Answering on TV Show Transcripts](https://www.aclweb.org/anthology/W19-5923). Zhengzhe Yang and Jinho D. Choi. In Proceedings of the Annual Conference of the ACL Special Interest Group on Discourse and Dialogue, SIGDIAL'19, 2019.

## Contact

* [Jinho D. Choi](http://www.mathcs.emory.edu/~choi).
