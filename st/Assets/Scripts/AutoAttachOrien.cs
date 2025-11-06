using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;

public class AutoAttachOrien : MonoBehaviour
{
    public XRGrabInteractable Bow;


    public Transform BowAttach;

    public XRDirectClimbInteractor temp = null;

    public void OnSelectEnter(SelectEnterEventArgs args)
    {
        temp = (XRDirectClimbInteractor)args.interactorObject;

        if(temp.CompareTag("RightHand"))
        {
            BowAttach.localPosition = new Vector3(0.01f, -0.06f, -0.065f);
            BowAttach.localRotation = Quaternion.Euler(0.0f, -21.2f, 0.0f);
        }

        if (temp.CompareTag("LeftHand"))
        {
            BowAttach.localPosition = new Vector3(-0.01f, -0.06f, -0.065f);
            BowAttach.localRotation = Quaternion.Euler(0.0f, 24.8f, 0.0f);

        }

    }




    /*    public EMGInteraction emgInteraction;

        public Transform BowAttach;



        // Start is called before the first frame update
        void Start()
        {

        }

        // Update is called once per frame
        void Update()
        {
            if(emgInteraction.CurrentHand == EMGInteraction.Hand.Right)
            {
                BowAttach.localPosition = new Vector3(0.0f, -0.0819f, -0.0542f);
                BowAttach.localRotation = Quaternion.Euler(0.0f, -21.2f, 0.0f);
                Debug.Log("for right");

            }
            if(emgInteraction.CurrentHand == EMGInteraction.Hand.Left)
            {
                BowAttach.localPosition = new Vector3(0.011f, 0.0817f, 0.0824f);
                BowAttach.localRotation = Quaternion.Euler(0.0f, 42.95f, 0.0f);
                Debug.Log("Fpr left");
            }
        }
    */


}
