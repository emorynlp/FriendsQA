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
The format of the data separates the context into several utterances and separates the speakers and utterance for each utterance. 

```json
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
                                "Carol Willick"
                            ],
                            "utterance": "You 're gon na kill me !"
                        }
  ]
```

The "qas" field includes questions and the answers. An answer has five keys. The first is "answer_text" which denotes the original text. The second is "utterance_id" which denotes the answer appearing in which utterance. The "inner_start" and "inner_end" denote the answer start and end token position in the corresponding utterance and if answer is the speaker,  their values are -1 and the "is_speaker" is set as true . 

```json
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
  }
  ]
```

## Citation

* [Transformers to Learn Hierarchical Contexts in Multiparty Dialogue for Span-based Question Answering](). Changmao Li and Jinho D. Choi. In Proceedings of the Conference of the Association for Computational Linguistics, ACL'20, 2020.
* [FriendsQA: Open-Domain Question Answering on TV Show Transcripts](https://www.aclweb.org/anthology/W19-5923). Zhengzhe Yang and Jinho D. Choi. In Proceedings of the Annual Conference of the ACL Special Interest Group on Discourse and Dialogue, SIGDIAL'19, 2019.

## Contact

* [Jinho D. Choi](http://www.mathcs.emory.edu/~choi).