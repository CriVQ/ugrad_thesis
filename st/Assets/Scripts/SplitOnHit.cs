using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;


public class SplitOnHit : MonoBehaviour
{
    // Reference to the AudioSource component
   // public AudioSource audioSource;

    // Reference to the VFX prefab
    public GameObject vfxPrefab;

    // The position at which the VFX should be displayed
    //public Vector3 vfxPosition;

    public XRGrabInteractable Club;

    // Reference to the prefab that will be used to create the split pieces
    public GameObject splitPrefab;

    // The number of pieces to split the object into
    public int numPieces = 5;

    // The force that will be applied to each split piece when it is created
    public float explosionForce = 10f;

    // The radius in which the force will be applied
    public float explosionRadius = 5f;

    // The upward modification to apply to the explosion force
    public float explosionUpwardModifier = 0.5f;

    [SerializeField] private GameObject ScoreCanvas;

    private ScoreManager scoremanager;
    
    
    void Start()
    {
        // Get a reference to the AudioSource component
        // audioSource = GetComponent<AudioSource>();
        scoremanager = ScoreCanvas.GetComponent<ScoreManager>();
        
    }


    void OnCollisionEnter(Collision collision)
    {
        // Check if the object that collided with this object has a tag of "Bullet"
        if (collision.collider.tag == "Club")
        {
            if (Club.isSelected)
            {
                // Split the object into pieces
                Split();

                // Add point in score manager script
                scoremanager.AddPoint();

                // Display the VFX
                DisplayVFX(transform.position, vfxPrefab.transform.rotation);

                // Play the collision sound
                // audioSource.Play();
            }
        }
    }

    void DisplayVFX(Vector3 vfxPosition, Quaternion vfxRotation)
    {
        // Instantiate the VFX prefab
        GameObject vfx = Instantiate(vfxPrefab, vfxPosition, vfxRotation);
    }


    void Split()
    {
        // Create an array to store the split pieces
        GameObject[] pieces = new GameObject[numPieces];

        // Loop through the array and create a new split piece for each iteration
        for (int i = 0; i < numPieces; i++)
        {
            // Instantiate a new split piece and store it in the array
            pieces[i] = Instantiate(splitPrefab, transform.position, transform.rotation);

            // Add a Rigidbody component to the split piece
            Rigidbody rb = pieces[i].AddComponent<Rigidbody>();
            rb.mass = 1;

            // Add a Box collider to the split piece
            BoxCollider collider = pieces[i].AddComponent<BoxCollider>();
            collider.size = new Vector3(0.05f, 0.05f, 0.05f);
        }

        // Get the position of the object
        Vector3 explosionPosition = transform.position;

        // Apply the explosion force to each split piece
        for (int i = 0; i < numPieces; i++)
        {
            Rigidbody rb = pieces[i].GetComponent<Rigidbody>();
            // rb.AddExplosionForce(explosionForce, explosionPosition, explosionRadius, explosionUpwardModifier);
            // Calculate a random force to apply in the upward direction
            Vector3 force = new Vector3(Random.Range(-1f, 1f), Random.Range(0.5f, 1f), Random.Range(-1f, 1f));
            force = force.normalized * explosionForce;

            rb.AddExplosionForce(explosionForce, explosionPosition, explosionRadius, explosionUpwardModifier);
            rb.AddForce(force, ForceMode.Impulse);
        }

        // Destroy the original object
        Destroy(gameObject);

        
    }

    
}