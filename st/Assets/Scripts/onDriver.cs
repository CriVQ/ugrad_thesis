using UnityEngine;

public class onDriver : MonoBehaviour
{

    public GameObject Cube;
    public Vector3 tempPos;

    public LiftingScoreManager _lift;


    [SerializeField]
    float eulerAngZ;


    void Update()
    {

        tempPos = Cube.transform.position;
        eulerAngZ = transform.localEulerAngles.z;

        tempPos.y = (eulerAngZ / 100) - 1.5f;

       
        Cube.transform.position = tempPos;

        if (Cube.transform.position.y < 0.35)
        {
            Destroy(GetComponent<SteeringWheel>());
            Destroy(GetComponent<HingeJoint>());
            Destroy(GetComponent<Rigidbody>());
            Destroy(GetComponent<onDriver>());
            _lift.Closed();

        }


    }
}