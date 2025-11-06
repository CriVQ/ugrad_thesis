using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Hit2Alien : MonoBehaviour
{

    [SerializeField] private Animator Warrior_Death;
    public SelfDefScoreManager _knightscore;


    private void OnTriggerEnter(Collider other)
    {

        {
            Destroy(this);
            Destroy(other.gameObject);

            _knightscore.AddPoint();

            Warrior_Death.SetBool("Hit2", true);
            //Destroy(GetComponent<Animator>());

        }
    }

}




