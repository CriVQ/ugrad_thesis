using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;


public class HandComplete : MonoBehaviour
{
    //Stores handPrefab to be Instantiated
    public GameObject handPrefab_Normal;
    public GameObject handPrefab_Robotic;

    //Allows for hiding of hand prefab if set to true
    public bool hideHandOnSelect = false;

    private bool _isHandHuman = true;

    //public EMGInteraction emginteraction;


    //Stores what kind of characteristics we're looking for with our Input Device when we search for it later
    public InputDeviceCharacteristics inputDeviceCharacteristics;

    //Stores the InputDevice that we're Targeting once we find it in InitializeHand()
    private InputDevice _targetDevice;
    [SerializeField] private Animator _handAnimator;
    private SkinnedMeshRenderer _handMesh;
    
    private GameObject spawnedHand;
    private GameObject dummy;

    private bool _isInitialized = false;

    public void HideHandOnSelect()
    {
        if (hideHandOnSelect)
        {
            _handMesh.enabled = !_handMesh.enabled;
        }
    }
    private void Start()
    {
        InitializeHand();
    }

    public void HideHandOnSelectEMG()
    {
        if (_handAnimator != null)
        {
            _handAnimator.SetBool("GripEMG", true);
        }
          
    }

    public void HideHandOnDeSelectEMG()
    {
        if (_handAnimator != null)
        {
            _handAnimator.SetBool("GripEMG", false);
        }

    }

    public void H()
    {

        Debug.Log("TestH");

    }


    

    public void InitializeHand()
    {
        List<InputDevice> devices = new List<InputDevice>();
        //Call InputDevices to see if it can find any devices with the characteristics we're looking for
        InputDevices.GetDevicesWithCharacteristics(inputDeviceCharacteristics, devices);

        //Our hands might not be active and so they will not be generated from the search.
        //We check if any devices are found here to avoid errors.
        if (devices.Count > 0)
        {
            
            _targetDevice = devices[0];

/*            spawnedHand = Instantiate(handPrefab_Normal, transform);
            _handAnimator = spawnedHand.GetComponent<Animator>();
            _handMesh = spawnedHand.GetComponentInChildren<SkinnedMeshRenderer>();

            _isHandHuman = true;

            dummy = spawnedHand;*/

            _isInitialized = true;
        }
        else if (devices.Count == 0)
        {
            _isInitialized = false;
        }
        
    }


    // Update is called once per frame
     void Update()
    {
        //Since our target device might not register at the start of the scene, we continously check until one is found.
        if(!_targetDevice.isValid)
        {
            InitializeHand();
            
        }
        else
        {
            UpdateHand();
        }
    }

    private void UpdateHand()
    {
        //This will get the value for our trigger from the target device and output a flaot into triggerValue


        if (_targetDevice.TryGetFeatureValue(CommonUsages.trigger, out float triggerValue))
        {
            _handAnimator.SetFloat("Trigger", triggerValue);
            

        }
        else
        {
            _handAnimator.SetFloat("Trigger", 0);
        }


        // _handMesh = spawnedHand.GetComponentInChildren<SkinnedMeshRenderer>();
        //This will get the value for our grip from the target device and output a flaot into gripValue
        /*

                if (_targetDevice.TryGetFeatureValue(CommonUsages.grip, out float gripValue))
                {
                    _handAnimator.SetFloat("Grip", gripValue);
                }
                else
                {
                    _handAnimator.SetFloat("Grip", 0);
                }

        */
    }

    public void SettoRobot()
    {


        {
            Destroy(dummy);
            spawnedHand = Instantiate(handPrefab_Robotic, transform);
            _handAnimator = spawnedHand.GetComponent<Animator>();
            _handMesh = spawnedHand.GetComponentInChildren<SkinnedMeshRenderer>();
            dummy = spawnedHand;

            _isHandHuman = false;
        }
    }

    public void SettoHuman() { 
                Destroy(dummy);
                spawnedHand = Instantiate(handPrefab_Normal, transform);
                _handAnimator = spawnedHand.GetComponent<Animator>();
                _handMesh = spawnedHand.GetComponentInChildren<SkinnedMeshRenderer>();
                dummy = spawnedHand;

                _isHandHuman = true;
            }
        }
    
