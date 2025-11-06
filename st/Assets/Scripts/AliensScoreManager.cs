using UnityEngine;
using UnityEngine.UI;

public class AliensScoreManager : MonoBehaviour
{
    public Text scoreText;


    int score = 0;

    public void Awake()
    {
        //  instance = this;
    }

    // Start is called before the first frame update
    void Start()
    {
        scoreText.text = score.ToString() + "/3 ALIENS";
    }

    // Update is called once per frame
    public void AddPoint()
    {
        score += 1;
        scoreText.text = score.ToString() + "/3 ALIENS";
    }


}
