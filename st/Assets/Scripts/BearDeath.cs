using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BearDeath : MonoBehaviour
{
    [SerializeField] private Animator Bear_Death;

    public BearScoreManager _bearScore;

    private void OnTriggerEnter(Collider other)
    {
       
        {
            if(other.gameObject.name == "Rock") { Destroy(other.gameObject); }
            
            
            Destroy(other.gameObject);
            Bear_Death.SetBool("Death", true);
            _bearScore.AddPoint();
            
            Destroy(this);
        }
    }

}
