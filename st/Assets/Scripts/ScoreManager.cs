using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ScoreManager : MonoBehaviour
{
    //public static ScoreManager instance;

    public Text scoreText;
    public Text scoreText2;

    int score = 0;

    public void Awake()
    {
        //  instance = this;
    }

    // Start is called before the first frame update
    void Start()
    {
        scoreText.text = score.ToString() + "/20 APPLES";
        scoreText2.text = score.ToString() + "/20 APPLES";
    }

    // Update is called once per frame
    public void AddPoint()
    {
        score += 1;
        scoreText.text = score.ToString() + "/20 APPLES";
        scoreText2.text = score.ToString() + "/20 APPLES";
    }
}
