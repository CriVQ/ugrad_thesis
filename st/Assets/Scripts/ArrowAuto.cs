using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;

public class ArrowAuto : MonoBehaviour
{

    public GameObject _arrow;
    public XRGrabInteractable _arrowXRComp;
    public GameObject notchpos;

    public XRSocketInteractor _notch;

    public bool arrowspawn = false;

    private void Awake()
    {
        
    }

    // public float timer = 0.0f;
    /*    void Update()
        {
            timer += Time.deltaTime;
            if (timer > 3.8)
            {
                Instantiate(_arrow, transform.position, Quaternion.Euler(0f, 220f, 0f));
                timer = 0;
            }
        }
    
       public void ArrowAttach()
    {
        Instantiate(_arrow, transform.position, Quaternion.Euler(0f, 220f, 0f));

    }
     
     */

    private void Update()
    {
        if (_notch.CanSelect(_arrowXRComp))
        {
            Instantiate(_arrow, transform.position, Quaternion.Euler(0f, 220f, 0f));

        }

        if (arrowspawn)
        {
            Instantiate(_arrow, transform.position, Quaternion.Euler(0f, 220f, 0f));
            arrowspawn = false;
        }
    }



}
