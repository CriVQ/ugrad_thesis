using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;

public class SmashSoundE : MonoBehaviour
{
    public AudioSource audioPlayer;
    public XRGrabInteractable Club;

    public void OnCollisionEnter(Collision collision)
    {
        if(collision.gameObject.tag == "Apple")
        {
            if (Club.isSelected)
            {
                audioPlayer.Play();
            }
            

        }
    }
}
