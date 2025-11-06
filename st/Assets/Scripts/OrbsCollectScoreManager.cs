using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.SceneManagement;

public class OrbsCollectScoreManager : MonoBehaviour
{
    public Text scoreText;
    public VolumeatStart fadeout;

    public XRBaseInteractable _temp;

    [SerializeField] public string NextSceneName;



    int score = 0;

    public void Awake()
    {
        //  instance = this;
    }

    // Start is called before the first frame update
    void Start()
    {
        scoreText.text = score.ToString() + "/7 ORBS";
    }

    // Update is called once per frame
    public void AddPoint(SelectEnterEventArgs args)
    {
        score += 1;
        scoreText.text = score.ToString() + "/7 ORBS";
        _temp = (XRBaseInteractable)args.interactableObject;
        Destroy(_temp.gameObject);
        _temp = null;
    }

    /*public void OnSelectDestroy()
    {
        Destroy(_rock);
    }*/
    private void Update()
    {
        if (score == 7 || score > 7)
        {
            fadeout.OnSceneChange();
            Invoke("LoadNextScene", 4);
        }
    }


    private void LoadNextScene()
    {
        SceneManager.LoadScene(NextSceneName);

    }
}
