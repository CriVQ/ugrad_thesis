using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ObjectExploder : MonoBehaviour
{
    public GameObject explosionPrefab; // prefab to use for the explosion effect
    public float explosionForce = 10.0f; // force of the explosion
    public float explosionRadius = 10.0f; // radius of the explosion

    // Use this for initialization
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    void OnTriggerEnter(Collider other)
    {
        if (other.tag != "Floor")
        {
            // Create the explosion effect
            Instantiate(explosionPrefab, transform.position, transform.rotation);

            // Apply explosion force to all nearby rigidbodies
            Collider[] colliders = Physics.OverlapSphere(transform.position, explosionRadius);
            foreach (Collider collider in colliders)
            {
                if (collider.attachedRigidbody != null && collider.tag != "Floor")
                {
                    collider.attachedRigidbody.AddExplosionForce(explosionForce, transform.position, explosionRadius);
                }
            }

            // Destroy the object
            Destroy(gameObject);
        }
    }
}