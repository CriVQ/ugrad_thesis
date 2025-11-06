using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChopSound : MonoBehaviour
{
    public AudioSource audioPlayer;

    private AxeandWoodScoreManager AxeandWoodScoremanager;

    [SerializeField] private GameObject ScoreCanvas;




    // Start is called before the first frame update
    void Start()
    {
        AxeandWoodScoremanager = ScoreCanvas.GetComponent<AxeandWoodScoreManager>();

    }

    // Update is called once per frame
    void Update()
    {

    }

    public void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.tag == "Log")

        {
            audioPlayer.Play();

            AxeandWoodScoremanager.AddPoint();
        }
    }
}
