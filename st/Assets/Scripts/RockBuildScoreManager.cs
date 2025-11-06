using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.Interaction.Toolkit;

public class RockBuildScoreManager : MonoBehaviour
{
    public Text scoreText;

    int score = 0;

    public void Awake()
    {
    }

    // Start is called before the first frame update
    void Start()
    {
        scoreText.text = score.ToString() + "/4 BRICKS";
    }

    // Update is called once per frame
    public void AddPoint()
    {
        score += 1;
        scoreText.text = score.ToString() + "/4 BRICKS";

    }

    /*public void OnSelectDestroy()
    {
        Destroy(_rock);
    }*/

}
