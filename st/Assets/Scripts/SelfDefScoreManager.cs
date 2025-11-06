
using UnityEngine;
using UnityEngine.UI;

public class SelfDefScoreManager : MonoBehaviour
{

    public GameObject RightGun;
    public GameObject LeftGun;
    public EMGInteraction _emg;

    public Text scoreText;


    int score = 0;

    public void Awake()
    {
        //  instance = this;
    }

    // Start is called before the first frame update
    void Start()
    {
        scoreText.text = score.ToString() + "/9 ALIENS";
        _emg._ManualAnimate = true;
    }

    // Update is called once per frame
    public void AddPoint()
    {
        score += 1;
        scoreText.text = score.ToString() + "/9 ALIENS";
    }

    private void Update()
    {
        if (_emg.CurrentHand == EMGInteraction.Hand.Right && !RightGun.activeSelf && RightGun && LeftGun)
        {
            RightGun.SetActive(true);
            LeftGun.SetActive(false);
            _emg.AnimateFist();
        }

        if (_emg.CurrentHand == EMGInteraction.Hand.Left && !LeftGun.activeSelf && RightGun && LeftGun)
        {
            LeftGun.SetActive(true);
            RightGun.SetActive(false);
            _emg.AnimateFist();
        }
    }
}
