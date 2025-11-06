using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class hit2 : MonoBehaviour
{

    [SerializeField] private Animator Warrior_Death;
    public KnightsScoreManager _knightscore;



    private void OnTriggerEnter(Collider other)
    {

        {
            other.gameObject.GetComponentInChildren<TrailRenderer>().enabled = false;

            _knightscore.AddPoint();

            Warrior_Death.SetBool("Hit2", true);
            //Destroy(GetComponent<Animator>());

        }
    }





}
