using UnityEngine;
using UnityEngine.UI;


public class LiftingScoreManager : MonoBehaviour
{
    public Text scoreText;

    // Start is called before the first frame update
    void Start()
    {
        scoreText.text =  "Turn Wheel Left";
    }


    public void Closed()
    {
        scoreText.text = "Gate is Closed";
    }


}
