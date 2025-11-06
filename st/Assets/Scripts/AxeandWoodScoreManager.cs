using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class AxeandWoodScoreManager : MonoBehaviour
{
    //public static ScoreManager instance;

    public Text AAWscoreText;

    public GameObject upperlog;

    int score = 0;

    public void Awake()
    {
        //  instance = this;
    }

    // Start is called before the first frame update
    void Start()
    {

        AAWscoreText.text = score.ToString() + "/5 HITS";
    }

    // Update is called once per frame
    public void AddPoint()
    {
        score += 1;
        AAWscoreText.text = score.ToString() + "/5 HITS";

        if (score >= 5)
        {
            upperlog.GetComponent<Rigidbody>().constraints = RigidbodyConstraints.None;
        }

    }



}
