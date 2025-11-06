using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FireBullet : MonoBehaviour
{

    public GameObject bulletPrefab;
    public Transform barrelLocation;
    public  float shotPower = 500.0f ;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {

       if (Input.GetKeyDown(KeyCode.Space))
        {
            Fire();
        }
    }
    
    public void Fire()
    {
        Instantiate(bulletPrefab, barrelLocation.position, barrelLocation.rotation).GetComponent<Rigidbody>().AddForce(barrelLocation.forward * shotPower);
    }



}
