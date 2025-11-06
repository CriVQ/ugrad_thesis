using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ObjectSplitter : MonoBehaviour
{
    public GameObject prefab; // prefab to use for the split pieces

    // Use this for initialization
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    void OnCollisionEnter(Collision collision)
    {
        // Split the object
        Vector3 position = transform.position;
        Quaternion rotation = transform.rotation;
        Vector3 scale = transform.localScale;
        GameObject piece1 = Instantiate(prefab, position, rotation);
        GameObject piece2 = Instantiate(prefab, position, rotation);
        piece1.transform.localScale = scale;
        piece2.transform.localScale = scale;
        Destroy(gameObject);
    }
}